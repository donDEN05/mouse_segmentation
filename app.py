"""
Flask приложение для обработки видео с использованием предобученных YOLO моделей
Позволяет загружать видео и получать обработанное видео с bounding boxes
"""
import os
import cv2
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import torch
from yolo_using import find_available_models, load_model

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Максимальный размер файла: 500MB
app.config['SECRET_KEY'] = os.urandom(24)

# Настройка путей
BASE_DIR = Path(__file__).parent.absolute()
RUNS_DIR = BASE_DIR / "runs" / "detect"
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"

# Создаем необходимые директории
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Разрешенные расширения видео файлов
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm', 'm4v'}


def allowed_file(filename):
    """Проверка, что файл имеет разрешенное расширение"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def discover_models():
    """
    Автоматический поиск всех доступных YOLO моделей из runs/detect
    Возвращает словарь с информацией о моделях для отображения в UI
    """
    models_dict = find_available_models(str(RUNS_DIR))
    
    # Преобразуем в формат для UI
    models_info = {}
    for exp_name, model_info in models_dict.items():
        available_versions = []
        if model_info["best"]:
            available_versions.append("best")
        if model_info["last"]:
            available_versions.append("last")
        
        if available_versions:
            # Формируем читаемое имя модели
            display_name = exp_name.replace('_', ' ').title()
            
            models_info[exp_name] = {
                'name': display_name,
                'versions': available_versions,
                'default_version': 'best' if 'best' in available_versions else available_versions[0]
            }
    
    return models_info


# Глобальный словарь для хранения загруженных моделей (кэширование)
loaded_models = {}

# Определение устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(experiment_name, version='best'):
    """
    Получить или загрузить YOLO модель для указанного эксперимента и версии
    
    Parameters:
    -----------
    experiment_name : str
        Имя эксперимента (название папки в runs/detect)
    version : str
        Версия весов: 'best' или 'last'
        
    Returns:
    --------
    YOLO model object или None в случае ошибки
    """
    model_key = f"{experiment_name}_{version}"
    
    if model_key not in loaded_models:
        try:
            model = load_model(experiment_name, version, str(RUNS_DIR))
            if model is not None:
                loaded_models[model_key] = model
                print(f"✓ Модель загружена: {model_key}")
            else:
                print(f"❌ Не удалось загрузить модель: {model_key}")
                return None
        except Exception as e:
            print(f"❌ Ошибка при загрузке модели {model_key}: {e}")
            return None
    
    return loaded_models[model_key]


def process_video_with_yolo(video_path, output_path, model, conf_threshold=0.25):
    """
    Обрабатывает видео с помощью YOLO модели и рисует bounding boxes
    
    Parameters:
    -----------
    video_path : str or Path
        Путь к входному видео файлу
    output_path : str or Path
        Путь для сохранения обработанного видео
    model : YOLO
        Загруженная YOLO модель
    conf_threshold : float
        Порог уверенности для детекции (по умолчанию 0.25)
        
    Returns:
    --------
    bool : True если обработка успешна, False иначе
    """
    try:
        # Открываем входное видео
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Не удалось открыть видео: {video_path}")
            return False
        
        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Параметры видео: {width}x{height}, {fps} FPS, {total_frames} кадров")
        
        # Создаем VideoWriter для выходного видео
        # Используем 'mp4v' codec (совместим с большинством браузеров через HTML5 video)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"❌ Не удалось создать выходное видео: {output_path}")
            print(f"   Попробуйте установить дополнительные кодеки для OpenCV")
            cap.release()
            return False
        
        frame_count = 0
        
        # Обрабатываем каждый кадр
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Применяем YOLO детекцию
            results = model(frame, conf=conf_threshold, verbose=False)
            
            # Рисуем bounding boxes на кадре
            annotated_frame = results[0].plot()
            
            # Записываем обработанный кадр
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Обработано кадров: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
        
        # Освобождаем ресурсы
        cap.release()
        out.release()
        
        print(f"✓ Видео успешно обработано: {frame_count} кадров")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при обработке видео: {e}")
        return False


@app.route('/')
def index():
    """Главная страница с формой загрузки видео"""
    models = discover_models()
    return render_template('index.html', models=models)


@app.route('/api/models', methods=['GET'])
def get_models():
    """API endpoint для получения списка доступных моделей"""
    models = discover_models()
    
    models_list = []
    for key, info in models.items():
        models_list.append({
            'key': key,
            'name': info['name'],
            'versions': info['versions'],
            'default_version': info['default_version']
        })
    
    return jsonify({'models': models_list})


@app.route('/api/process_video', methods=['POST'])
def process_video():
    """API endpoint для обработки видео"""
    try:
        # Проверяем наличие файла
        if 'video' not in request.files:
            return jsonify({'error': 'Видео файл не загружен'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400
        
        if not allowed_file(video_file.filename):
            return jsonify({
                'error': f'Неподдерживаемый формат файла. Разрешенные форматы: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Получаем параметры запроса
        model_key = request.form.get('model')
        version = request.form.get('version', 'best')
        conf_threshold = float(request.form.get('conf_threshold', 0.25))
        
        if not model_key:
            return jsonify({'error': 'Модель не выбрана'}), 400
        
        # Проверяем наличие модели
        models = discover_models()
        if model_key not in models:
            return jsonify({'error': f'Модель "{model_key}" не найдена'}), 400
        
        if version not in models[model_key]['versions']:
            return jsonify({
                'error': f'Версия "{version}" недоступна для модели "{model_key}"'
            }), 400
        
        # Загружаем модель
        model = get_model(model_key, version)
        if model is None:
            return jsonify({'error': f'Не удалось загрузить модель {model_key}/{version}'}), 500
        
        # Сохраняем загруженное видео
        filename = secure_filename(video_file.filename)
        input_video_path = UPLOAD_FOLDER / filename
        video_file.save(str(input_video_path))
        
        # Формируем имя выходного файла
        output_filename = f"processed_{Path(filename).stem}.mp4"
        output_video_path = OUTPUT_FOLDER / output_filename
        
        # Обрабатываем видео
        print(f"\n{'='*60}")
        print(f"🎬 НАЧАЛО ОБРАБОТКИ ВИДЕО")
        print(f"{'='*60}")
        print(f"Входной файл: {input_video_path}")
        print(f"Выходной файл: {output_video_path}")
        print(f"Модель: {model_key}/{version}")
        print(f"Устройство: {device}")
        print(f"Порог уверенности: {conf_threshold}")
        
        success = process_video_with_yolo(
            input_video_path,
            output_video_path,
            model,
            conf_threshold=conf_threshold
        )
        
        if not success:
            # Удаляем входной файл при ошибке
            if input_video_path.exists():
                input_video_path.unlink()
            return jsonify({'error': 'Ошибка при обработке видео'}), 500
        
        # Удаляем входной файл после успешной обработки
        if input_video_path.exists():
            input_video_path.unlink()
        
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'message': 'Видео успешно обработано',
            'output_file': output_filename,
            'model': model_key,
            'version': version
        })
    
    except ValueError as e:
        return jsonify({'error': f'Некорректное значение параметра: {str(e)}'}), 400
    except Exception as e:
        print(f"❌ Ошибка при обработке запроса: {e}")
        return jsonify({'error': f'Ошибка при обработке: {str(e)}'}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_video(filename):
    """API endpoint для скачивания обработанного видео"""
    try:
        file_path = OUTPUT_FOLDER / secure_filename(filename)
        
        if not file_path.exists():
            return jsonify({'error': 'Файл не найден'}), 404
        
        return send_file(
            str(file_path),
            as_attachment=True,
            download_name=filename,
            mimetype='video/mp4'
        )
    except Exception as e:
        return jsonify({'error': f'Ошибка при загрузке файла: {str(e)}'}), 500


@app.route('/api/video/<filename>', methods=['GET'])
def stream_video(filename):
    """API endpoint для потоковой передачи обработанного видео (для HTML5 video player)"""
    try:
        file_path = OUTPUT_FOLDER / secure_filename(filename)
        
        if not file_path.exists():
            return jsonify({'error': 'Файл не найден'}), 404
        
        # Используем send_file с conditional=True для поддержки range requests
        # Это необходимо для правильного воспроизведения видео в HTML5 video player
        return send_file(
            str(file_path),
            mimetype='video/mp4',
            conditional=True,  # Включает поддержку HTTP Range requests
            as_attachment=False
        )
        
    except Exception as e:
        return jsonify({'error': f'Ошибка при загрузке файла: {str(e)}'}), 500


if __name__ == '__main__':
    print("="*60)
    print("🚀 ЗАПУСК YOLO VIDEO PROCESSING API")
    print("="*60)
    print(f"🖥️  Используемое устройство: {device}")
    print(f"📁 Директория моделей: {RUNS_DIR}")
    print(f"📤 Директория загрузок: {UPLOAD_FOLDER}")
    print(f"📥 Директория результатов: {OUTPUT_FOLDER}")
    
    models = discover_models()
    if models:
        print(f"\n📦 Найдено моделей: {len(models)}")
        for key, info in models.items():
            versions_str = ', '.join(info['versions'])
            print(f"   ✓ {key} ({versions_str}) - {info['name']}")
    else:
        print("\n⚠️  Модели не найдены!")
        print(f"   Убедитесь, что в папке {RUNS_DIR} есть модели с весами в папке weights/")
    
    print("="*60)
    print(f"\n🌐 Сервер запущен на http://localhost:5000\n")
    
    # Запуск Flask приложения
    app.run(debug=True, host='0.0.0.0', port=5000)

