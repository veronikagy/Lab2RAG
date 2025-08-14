import streamlit as st
import asyncio
import logging
from typing import Optional
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from config import config
from src.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="RAG Система",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.section-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #333;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

.stats-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}

.error-box {
    background-color: #ffebee;
    color: #c62828;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #c62828;
}

.success-box {
    background-color: #e8f5e8;
    color: #2e7d32;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #2e7d32;
}

.context-chunk {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    border-left: 3px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'available_models' not in st.session_state:
    st.session_state.available_models = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

async def initialize_rag_pipeline():
    """Initialize RAG pipeline"""
    try:
        if not config.validate():
            st.error("Пожалуйста, проверьте конфигурацию в файле .env")
            st.stop()
        
        with st.spinner("Инициализация RAG системы..."):
            pipeline = RAGPipeline(
                openrouter_api_key=config.OPENROUTER_API_KEY,
                qdrant_url=config.QDRANT_URL,
                qdrant_api_key=config.QDRANT_API_KEY,
                collection_name=config.QDRANT_COLLECTION_NAME,
                embedding_model=config.EMBEDDING_MODEL,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            
            # Get available models
            models = await pipeline.get_available_models()
            
            return pipeline, models
            
    except Exception as e:
        st.error(f"Ошибка инициализации: {str(e)}")
        return None, []

def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🤖 RAG Система</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize pipeline if needed
    if st.session_state.rag_pipeline is None:
        pipeline, models = asyncio.run(initialize_rag_pipeline())
        if pipeline:
            st.session_state.rag_pipeline = pipeline
            st.session_state.available_models = models
            st.success("RAG система успешно инициализирована!")
        else:
            st.stop()
    
    # Sidebar for configuration and file upload
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Настройки</div>', unsafe_allow_html=True)
        
        # Model selection
        if st.session_state.available_models:
            model_options = [(m["id"], f"{m['name']} ({m['id']})") for m in st.session_state.available_models]
            selected_model_id = st.selectbox(
                "Выберите модель:",
                options=[opt[0] for opt in model_options],
                format_func=lambda x: next((opt[1] for opt in model_options if opt[0] == x), x),
                index=0
            )
        else:
            st.error("Не удалось загрузить список моделей")
            selected_model_id = "openai/gpt-3.5-turbo"
        
        # RAG settings
        st.markdown("**Настройки RAG:**")
        use_rag = st.checkbox("Использовать RAG", value=True)
        max_chunks = st.slider("Максимум чанков", 1, 10, 5)
        similarity_threshold = st.slider("Порог схожести", 0.0, 1.0, 0.1, 0.05)
        
        # Source filter
        sources = st.session_state.rag_pipeline.get_document_sources()
        source_filter = None
        if sources:
            use_source_filter = st.checkbox("Фильтр по документу")
            if use_source_filter:
                source_filter = st.selectbox("Выберите документ:", sources)
        
        st.markdown("---")
        
        # File upload section
        st.markdown('<div class="section-header">📄 Загрузка PDF</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Выберите PDF файл:",
            type=['pdf'],
            help="Загрузите PDF файл для добавления в базу знаний"
        )
        
        chunking_strategy = st.selectbox(
            "Стратегия разбиения:",
            ["fixed_size", "semantic"],
            format_func=lambda x: "Фиксированный размер" if x == "fixed_size" else "Семантические секции"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Обработать PDF", type="primary"):
                with st.spinner("Обработка PDF файла..."):
                    file_content = uploaded_file.read()
                    result = asyncio.run(
                        st.session_state.rag_pipeline.process_pdf(
                            file_content, 
                            uploaded_file.name,
                            chunking_strategy
                        )
                    )
                    
                    if result["success"]:
                        st.success(f"✅ Файл {uploaded_file.name} успешно обработан!")
                        st.json(result)
                    else:
                        st.error(f"❌ Ошибка: {result['error']}")
        
        st.markdown("---")
        
        # Document management
        st.markdown('<div class="section-header">📚 Управление документами</div>', unsafe_allow_html=True)
        
        if sources:
            st.write(f"**Документов в базе:** {len(sources)}")
            for source in sources:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(source)
                with col2:
                    if st.button("🗑️", key=f"delete_{source}", help="Удалить документ"):
                        if st.session_state.rag_pipeline.delete_document(source):
                            st.success(f"Документ {source} удален")
                            st.rerun()
                        else:
                            st.error("Ошибка удаления")
        else:
            st.info("Нет загруженных документов")
        
        # System stats
        if st.button("📊 Статистика системы"):
            stats = st.session_state.rag_pipeline.get_stats()
            st.json(stats)
    
    # Main chat interface
    st.markdown('<div class="section-header">💬 Чат с RAG системой</div>', unsafe_allow_html=True)
    
    # Chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Show context if available
                if "context_chunks" in message and message["context_chunks"]:
                    with st.expander("📖 Использованный контекст"):
                        for i, chunk in enumerate(message["context_chunks"]):
                            st.markdown(f"""
                            <div class="context-chunk">
                                <strong>Источник:</strong> {chunk.get('source', 'Неизвестно')}<br>
                                <strong>Релевантность:</strong> {chunk.get('score', 0):.3f}<br>
                                <strong>Текст:</strong> {chunk.get('text', '')[:300]}...
                            </div>
                            """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Задайте вопрос..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Генерация ответа..."):
                result = asyncio.run(
                    st.session_state.rag_pipeline.query(
                        question=prompt,
                        model=selected_model_id,
                        max_chunks=max_chunks,
                        similarity_threshold=similarity_threshold,
                        source_filter=source_filter,
                        use_rag=use_rag
                    )
                )
                
                if result["success"]:
                    st.write(result["answer"])
                    
                    # Add to history with context
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": result["answer"],
                        "context_chunks": result.get("context_chunks", []),
                        "rag_used": result.get("rag_used", False)
                    })
                    
                    # Show context
                    if result.get("context_chunks"):
                        with st.expander("📖 Использованный контекст"):
                            for i, chunk in enumerate(result["context_chunks"]):
                                st.markdown(f"""
                                <div class="context-chunk">
                                    <strong>Источник:</strong> {chunk.get('source', 'Неизвестно')}<br>
                                    <strong>Релевантность:</strong> {chunk.get('score', 0):.3f}<br>
                                    <strong>Текст:</strong> {chunk.get('text', '')[:300]}...
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Show RAG status
                    if result.get("rag_used"):
                        st.info("✅ Ответ сгенерирован с использованием RAG")
                    else:
                        st.warning("⚠️ Ответ сгенерирован без контекста из документов")
                        
                else:
                    error_msg = f"Ошибка: {result.get('error', 'Неизвестная ошибка')}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Clear chat button
    if st.button("🗑️ Очистить чат"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()
