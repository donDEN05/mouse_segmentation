"""
Flask приложение для обработки видео с использованием предобученных YOLO моделей
Позволяет загружать видео и получать обработанное видео с bounding boxes
"""
import os
import cv2
import json
import zipfile
import shutil
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import torch
from yolo_using import find_available_models, load_model

# Целевой размер для видео и изображений
TARGET_WIDTH = 1280
TARGET_HEIGHT = 960

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


def resize_video(video_path, output_path, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT):
    """
    Изменяет размер видео до указанного размера с сохранением пропорций (letterboxing)
    Видео масштабируется с сохранением пропорций, черные полосы добавляются при необходимости
    
    Parameters:
    -----------
    video_path : str or Path
        Путь к входному видео
    output_path : str or Path
        Путь для сохранения измененного видео
    target_width : int
        Целевая ширина (по умолчанию 1280)
    target_height : int
        Целевая высота (по умолчанию 960)
        
    Returns:
    --------
    bool : True если успешно, False иначе
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Не удалось открыть видео: {video_path}")
            return False
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Если размер уже правильный, просто копируем файл
        if width == target_width and height == target_height:
            shutil.copy2(video_path, output_path)
            cap.release()
            return True
        
        print(f"📐 Масштабирование видео с сохранением пропорций: {width}x{height} → {target_width}x{target_height}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))
        
        if not out.isOpened():
            print(f"❌ Не удалось создать выходное видео")
            cap.release()
            return False
        
        # Вычисляем коэффициенты масштабирования для сохранения пропорций
        scale_width = target_width / width
        scale_height = target_height / height
        scale = min(scale_width, scale_height)  # Используем меньший коэффициент, чтобы не обрезать
        
        # Вычисляем новые размеры после масштабирования
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Вычисляем смещения для центрирования
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        print(f"  Масштаб: {scale:.4f}, новый размер кадра: {new_width}x{new_height}, отступы: x={x_offset}, y={y_offset}")
        
        frame_count = 0
        background_color = (0, 0, 0)  # Черный фон для letterboxing
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Масштабируем кадр с сохранением пропорций
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Создаем кадр целевого размера с черным фоном
            result_frame = np.full((target_height, target_width, 3), background_color, dtype=np.uint8)
            
            # Вставляем масштабированный кадр в центр
            result_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
            
            out.write(result_frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Обработано кадров: {frame_count}/{total_frames}")
        
        cap.release()
        out.release()
        
        print(f"✓ Видео масштабировано с сохранением пропорций: {frame_count} кадров")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при изменении размера видео: {e}")
        return False


def process_video_with_yolo(video_path, output_path, model, conf_threshold=0.25):
    """
    Обрабатывает видео с помощью YOLO модели и рисует bounding boxes
    
    Parameters:
    -----------
    video_path : str or Path
        Путь к входному видео файлу (уже должен быть размера 1280x960)
    output_path : str or Path
        Путь для сохранения обработанного видео
    model : YOLO
        Загруженная YOLO модель
    conf_threshold : float
        Порог уверенности для детекции (по умолчанию 0.25)
        
    Returns:
    --------
    tuple : (bool, list) - (True если успешно, список аннотаций для metadata.json)
            Аннотации в формате: [{"frame_id": int, "annotations": [...]}, ...]
    """
    annotations_list = []
    
    try:
        # Открываем входное видео
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Не удалось открыть видео: {video_path}")
            return False, []
        
        # Получаем параметры видео (должны быть 1280x960)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Параметры видео: {width}x{height}, {fps} FPS, {total_frames} кадров")
        
        # Создаем VideoWriter для выходного видео
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"❌ Не удалось создать выходное видео: {output_path}")
            cap.release()
            return False, []
        
        frame_count = 0
        
        # Обрабатываем каждый кадр
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Применяем YOLO детекцию
            results = model(frame, conf=conf_threshold, verbose=False)
            
            # Собираем информацию о детекциях для metadata.json
            frame_annotations = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    # Получаем координаты bbox (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    
                    # Получаем category_id
                    category_id = int(box.cls[0].cpu().numpy())
                    
                    # Получаем confidence
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Формат bbox: [[x1, y1], [x2, y2]]
                    bbox = [[x1, y1], [x2, y2]]
                    
                    annotation = {
                        "bbox": bbox,
                        "bbox_mode": "BoxMode.XYXY_ABS",
                        "category_id": category_id,
                        "confidence": confidence
                    }
                    
                    frame_annotations.append(annotation)
            
            # Добавляем аннотации кадра в общий список
            annotations_list.append({
                "frame_id": frame_count,
                "annotations": frame_annotations
            })
            
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
        return True, annotations_list
        
    except Exception as e:
        print(f"❌ Ошибка при обработке видео: {e}")
        return False, []


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
        original_input_path = UPLOAD_FOLDER / filename
        video_file.save(str(original_input_path))
        
        # Проверяем размер видео и изменяем при необходимости
        cap = cv2.VideoCapture(str(original_input_path))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Путь для видео с правильным размером (используется для обработки)
        input_video_path = original_input_path
        needs_resize = (original_width != TARGET_WIDTH or original_height != TARGET_HEIGHT)
        
        if needs_resize:
            print(f"📐 Размер видео {original_width}x{original_height} не соответствует {TARGET_WIDTH}x{TARGET_HEIGHT}")
            print(f"   Изменение размера...")
            resized_input_path = UPLOAD_FOLDER / f"resized_{filename}"
            if resize_video(original_input_path, resized_input_path):
                input_video_path = resized_input_path
            else:
                if original_input_path.exists():
                    original_input_path.unlink()
                return jsonify({'error': 'Ошибка при изменении размера видео'}), 500
        
        # Формируем имена файлов
        base_name = Path(filename).stem
        output_filename = f"processed_{base_name}.mp4"
        output_video_path = OUTPUT_FOLDER / output_filename
        
        # Имя для исходного видео (сохраняем с правильным размером)
        source_filename = f"source_{base_name}.mp4"
        source_video_path = OUTPUT_FOLDER / source_filename
        
        # Копируем исходное видео (с правильным размером) в папку outputs
        shutil.copy2(input_video_path, source_video_path)
        
        # Обрабатываем видео
        print(f"\n{'='*60}")
        print(f"🎬 НАЧАЛО ОБРАБОТКИ ВИДЕО")
        print(f"{'='*60}")
        print(f"Входной файл: {input_video_path}")
        print(f"Выходной файл: {output_video_path}")
        print(f"Модель: {model_key}/{version}")
        print(f"Устройство: {device}")
        print(f"Порог уверенности: {conf_threshold}")
        
        success, annotations_list = process_video_with_yolo(
            input_video_path,
            output_video_path,
            model,
            conf_threshold=conf_threshold
        )
        
        if not success:
            # Удаляем файлы при ошибке
            if original_input_path.exists():
                original_input_path.unlink()
            if input_video_path != original_input_path and input_video_path.exists():
                input_video_path.unlink()
            if source_video_path.exists():
                source_video_path.unlink()
            return jsonify({'error': 'Ошибка при обработке видео'}), 500
        
        # Сохраняем metadata.json
        metadata_filename = f"metadata_{base_name}.json"
        metadata_path = OUTPUT_FOLDER / metadata_filename
        
        metadata = {
            "video_name": base_name,
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
            "annotations": annotations_list
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Metadata сохранен: {metadata_path}")
        
        # Удаляем временные входные файлы
        if original_input_path.exists():
            original_input_path.unlink()
        if input_video_path != original_input_path and input_video_path.exists():
            input_video_path.unlink()
        
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'message': 'Видео успешно обработано',
            'source_file': source_filename,
            'output_file': output_filename,
            'metadata_file': metadata_filename,
            'model': model_key,
            'version': version
        })
    
    except ValueError as e:
        return jsonify({'error': f'Некорректное значение параметра: {str(e)}'}), 400
    except Exception as e:
        print(f"❌ Ошибка при обработке запроса: {e}")
        return jsonify({'error': f'Ошибка при обработке: {str(e)}'}), 500


@app.route('/api/download/<base_name>', methods=['GET'])
def download_results(base_name):
    """
    API endpoint для скачивания всех результатов обработки в виде ZIP архива
    Включает: исходное видео, обработанное видео и metadata.json
    """
    try:
        # Безопасное имя файла
        safe_base_name = secure_filename(base_name)
        
        # Убираем префиксы если они есть
        if safe_base_name.startswith('processed_'):
            safe_base_name = safe_base_name.replace('processed_', '')
        if safe_base_name.startswith('source_'):
            safe_base_name = safe_base_name.replace('source_', '')
        if safe_base_name.startswith('metadata_'):
            safe_base_name = safe_base_name.replace('metadata_', '').replace('.json', '')
        
        # Имена файлов
        source_filename = f"source_{safe_base_name}.mp4"
        output_filename = f"processed_{safe_base_name}.mp4"
        metadata_filename = f"metadata_{safe_base_name}.json"
        
        source_path = OUTPUT_FOLDER / source_filename
        output_path = OUTPUT_FOLDER / output_filename
        metadata_path = OUTPUT_FOLDER / metadata_filename
        
        # Проверяем наличие всех файлов
        if not source_path.exists():
            return jsonify({'error': f'Исходное видео не найдено: {source_filename}'}), 404
        if not output_path.exists():
            return jsonify({'error': f'Обработанное видео не найдено: {output_filename}'}), 404
        if not metadata_path.exists():
            return jsonify({'error': f'Metadata файл не найден: {metadata_filename}'}), 404
        
        # Создаем временный ZIP архив
        zip_filename = f"{safe_base_name}_results.zip"
        zip_path = OUTPUT_FOLDER / zip_filename
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(source_path, source_filename)
            zipf.write(output_path, output_filename)
            zipf.write(metadata_path, metadata_filename)
        
        # Отправляем архив
        response = send_file(
            str(zip_path),
            as_attachment=True,
            download_name=zip_filename,
            mimetype='application/zip'
        )
        
        return response
        
    except Exception as e:
        print(f"❌ Ошибка при создании архива: {e}")
        return jsonify({'error': f'Ошибка при создании архива: {str(e)}'}), 500


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

