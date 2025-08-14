#!/bin/bash

echo "🚀 Запуск RAG системы"
echo "===================="

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден. Установите Python 3.8+"
    exit 1
fi

# Проверка pip
if ! command -v pip &> /dev/null; then
    echo "❌ pip не найден. Установите pip"
    exit 1
fi

echo "✅ Python найден: $(python3 --version)"

# Проверка .env файла
if [ ! -f .env ]; then
    echo "⚠️  Файл .env не найден!"
    echo "📝 Создайте .env файл на основе env.example"
    echo "💡 Скопируйте: cp env.example .env"
    echo "✏️  Затем отредактируйте .env и добавьте ваши API ключи"
    exit 1
fi

echo "✅ Файл .env найден"

# Установка зависимостей
echo "📦 Проверка зависимостей..."
pip install -q -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Ошибка установки зависимостей"
    exit 1
fi

echo "✅ Зависимости установлены"

# Создание тестового PDF если его нет
if [ ! -f "simple_test.pdf" ] && [ ! -f "test_tz_frontend.pdf" ]; then
    echo "📄 Создание тестового PDF..."
    python3 create_test_pdf.py
fi

# Меню выбора режима запуска
echo ""
echo "Выберите режим запуска:"
echo "1) 🌐 Веб-интерфейс (Streamlit)"
echo "2) 💻 Консольный режим"
echo "3) 🧪 Только тестирование подключений"

read -p "Введите номер (1-3): " choice

case $choice in
    1)
        echo "🌐 Запуск веб-интерфейса..."
        echo "🔗 Откроется по адресу: http://localhost:8501"
        streamlit run app.py
        ;;
    2)
        echo "💻 Запуск консольного режима..."
        python3 run.py
        ;;
    3)
        echo "🧪 Тестирование системы..."
        python3 -c "
import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path('.') / 'src'))
from config import config

async def quick_test():
    print('🔍 Проверка конфигурации...')
    if config.validate():
        print('✅ Конфигурация корректна')
        
        try:
            from src.rag_pipeline import RAGPipeline
            pipeline = RAGPipeline(
                openrouter_api_key=config.OPENROUTER_API_KEY,
                qdrant_url=config.QDRANT_URL,
                qdrant_api_key=config.QDRANT_API_KEY,
                collection_name=config.QDRANT_COLLECTION_NAME
            )
            health = pipeline.health_check()
            print(f'🏥 Статус компонентов: {health}')
            
            models = await pipeline.get_available_models()
            print(f'🤖 Доступно моделей: {len(models)}')
            print('✅ Система готова к работе!')
            
        except Exception as e:
            print(f'❌ Ошибка инициализации: {e}')
    else:
        print('❌ Проверьте настройки в .env файле')

asyncio.run(quick_test())
        "
        ;;
    *)
        echo "❌ Неверный выбор"
        exit 1
        ;;
esac
