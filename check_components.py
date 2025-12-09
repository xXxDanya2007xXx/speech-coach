#!/usr/bin/env python3
"""
Быстрая проверка компонентов системы.
"""

import sys
from pathlib import Path
import subprocess
import tempfile


def check_ffmpeg():
    """Проверяет наличие и работоспособность ffmpeg"""
    print("🔍 Проверяем FFmpeg...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ FFmpeg найден: {result.stdout.split('version')[0].strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg не найден или не работает")
        return False


def check_whisper():
    """Проверяет возможность загрузки модели Whisper"""
    print("\n🔍 Проверяем Whisper...")
    try:
        from faster_whisper import WhisperModel
        print("✅ Библиотека faster-whisper доступна")

        # Пробуем загрузить tiny модель для теста
        print("   Пробуем загрузить tiny модель (это может занять время)...")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✅ Tiny модель загружена успешно")
        return True
    except ImportError as e:
        print(f"❌ Библиотека faster-whisper не установлена: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return False


def check_gigachat_config():
    """Проверяет конфигурацию GigaChat"""
    print("\n🔍 Проверяем конфигурацию GigaChat...")
    try:
        from app.core.config import settings
        print(f"✅ Конфигурация загружена")
        print(f"   GigaChat включен: {settings.gigachat_enabled}")
        print(f"   API ключ установлен: {
              settings.gigachat_api_key is not None}")
        return True
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        return False


def create_test_audio():
    """Создаёт тестовый аудиофайл для проверки"""
    print("\n🔍 Создаём тестовый аудиофайл...")
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            test_audio = Path(tmp.name)

            # Создаём простой аудиофайл с помощью ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", "sine=frequency=1000:duration=2",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-hide_banner",
                "-loglevel", "error",
                str(test_audio)
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10)

            if test_audio.exists() and test_audio.stat().st_size > 0:
                print(f"✅ Тестовый аудиофайл создан: {
                      test_audio} ({test_audio.stat().st_size} bytes)")
                return test_audio
            else:
                print(f"❌ Не удалось создать тестовый аудиофайл")
                return None

    except Exception as e:
        print(f"❌ Ошибка создания тестового аудио: {e}")
        return None


def test_transcription():
    """Тестирует транскрипцию"""
    print("\n🔍 Тестируем транскрипцию...")
    try:
        from app.services.transcriber import LocalWhisperTranscriber

        # Создаём тестовый аудиофайл
        test_audio = create_test_audio()
        if not test_audio:
            return False

        # Пробуем транскрибировать
        transcriber = LocalWhisperTranscriber(model_size="tiny")
        transcript = transcriber.transcribe(test_audio)

        print(f"✅ Транскрипция успешна")
        print(f"   Сегментов: {len(transcript.segments)}")
        print(f"   Текст: {transcript.text[:100]
              if transcript.text else 'пусто'}...")

        # Удаляем временный файл
        test_audio.unlink()
        return True

    except Exception as e:
        print(f"❌ Ошибка транскрипции: {e}")
        return False


def main():
    """Основная функция проверки"""
    print("=" * 60)
    print("Speech Coach - Проверка компонентов")
    print("=" * 60)

    # Добавляем текущую директорию в путь Python
    sys.path.insert(0, str(Path(__file__).parent))

    checks = [
        ("FFmpeg", check_ffmpeg),
        ("Конфигурация", check_gigachat_config),
        ("Whisper библиотека", check_whisper),
        ("Транскрипция", test_transcription),
    ]

    results = []
    for name, check_func in checks:
        try:
            success = check_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ Ошибка при проверке {name}: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Результаты проверки:")
    print("=" * 60)

    all_passed = True
    for name, success in results:
        status = "✅ ПРОЙДЕНО" if success else "❌ НЕ ПРОЙДЕНО"
        print(f"{name}: {status}")
        if not success:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ Все проверки пройдены! Система готова к работе.")
    else:
        print("⚠️  Некоторые проверки не пройдены. Возможны проблемы.")
        print("\nРекомендации:")
        print("1. Убедитесь, что ffmpeg установлен и доступен в PATH")
        print("2. Проверьте установку faster-whisper: pip install faster-whisper")
        print("3. Для больших моделей Whisper потребуется больше времени на загрузку")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
