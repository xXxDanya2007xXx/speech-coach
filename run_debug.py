#!/usr/bin/env python3
"""
Запуск сервера в режиме отладки.
"""

import subprocess
import sys
import signal
import time


def main():
    """Запускает сервер с увеличенным уровнем логирования"""
    print("🚀 Запуск Speech Coach API в режиме отладки")
    print("=" * 60)

    # Проверяем наличие зависимостей
    try:
        import uvicorn
        import fastapi
        print("✅ Зависимости загружены")
    except ImportError as e:
        print(f"❌ Отсутствуют зависимости: {e}")
        print("Установите зависимости: pip install -r requirements.txt")
        return 1

    # Запускаем сервер
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--reload",
        "--port", "8000",
        "--host", "127.0.0.1",
        "--log-level", "debug"
    ]

    print(f"Команда: {' '.join(cmd)}")
    print("\nЛоги будут выводиться ниже. Нажмите Ctrl+C для остановки.")
    print("=" * 60)

    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        print("\n\n⏹️  Останавливаем сервер...")
        process.terminate()
        process.wait()
        print("✅ Сервер остановлен")

    return 0


if __name__ == "__main__":
    sys.exit(main())
