"""
Скрипт для запуска локального сервера MLflow UI
"""
import os
import subprocess
import sys

# Получаем базовую директорию проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

# Проверяем существование директории mlruns
if not os.path.exists(MLRUNS_DIR):
    print(f"⚠️  Директория mlruns не найдена. Создаю: {MLRUNS_DIR}")
    os.makedirs(MLRUNS_DIR, exist_ok=True)

print("="*60)
print("🚀 ЗАПУСК MLFLOW UI")
print("="*60)
print(f"📁 Директория mlruns: {MLRUNS_DIR}")
print(f"🌐 MLflow UI будет доступен по адресу: http://localhost:5000")
print("="*60)
print("\nДля остановки сервера нажмите Ctrl+C\n")

try:
    # Формируем правильный URI для Windows
    if os.name == 'nt':  # Windows
        # На Windows используем формат file:///D:/path/to/mlruns (с заглавной буквой диска)
        abs_path = os.path.abspath(MLRUNS_DIR)
        # Преобразуем путь: D:\path\to\mlruns -> D:/path/to/mlruns
        uri_path = abs_path.replace('\\', '/')
        # Убеждаемся, что буква диска заглавная
        if len(uri_path) > 1 and uri_path[1] == ':':
            uri_path = uri_path[0].upper() + uri_path[1:]
        backend_uri = f"file:///{uri_path}"
    else:  # Unix/Linux/Mac
        backend_uri = f"file://{os.path.abspath(MLRUNS_DIR)}"
    
    print(f"🔗 Backend URI: {backend_uri}")
    print()
    
    # Запускаем MLflow UI с указанием пути к mlruns
    subprocess.run(
        ["mlflow", "ui", "--backend-store-uri", backend_uri, "--port", "5000"],
        check=True
    )
except KeyboardInterrupt:
    print("\n\n✅ MLflow UI остановлен")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Ошибка при запуске MLflow UI: {e}")
    print("\nПопробуйте запустить вручную:")
    if os.name == 'nt':
        abs_path = os.path.abspath(MLRUNS_DIR)
        uri_path = abs_path.replace('\\', '/')
        if len(uri_path) > 1 and uri_path[1] == ':':
            uri_path = uri_path[0].upper() + uri_path[1:]
        backend_uri = f"file:///{uri_path}"
    else:
        backend_uri = f"file://{os.path.abspath(MLRUNS_DIR)}"
    print(f"   mlflow ui --backend-store-uri {backend_uri} --port 5000")
    sys.exit(1)
except FileNotFoundError:
    print("\n❌ MLflow не найден. Убедитесь, что MLflow установлен:")
    print("   pip install mlflow")
    sys.exit(1)

