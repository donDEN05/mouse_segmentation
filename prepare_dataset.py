"""
Скрипт для подготовки датасета из видео для дообучения YOLO11
Извлекает кадры из видео и конвертирует аннотации в формат YOLO
"""
import json
import cv2
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Конвертирует bbox из формата [[x1, y1], [x2, y2]] в формат YOLO
    
    Parameters:
    -----------
    bbox : list
        [[x1, y1], [x2, y2]] - координаты в абсолютных пикселях
    img_width : int
        Ширина изображения
    img_height : int
        Высота изображения
        
    Returns:
    --------
    tuple : (x_center, y_center, width, height) - нормализованные значения 0-1
    """
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    
    # Вычисление центра и размеров
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Нормализация в диапазон [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height


def extract_frames_and_annotations(video_path, metadata_path, output_dir, 
                                   target_width=1280, target_height=960):
    """
    Извлекает кадры из видео и создает аннотации в формате YOLO
    
    Parameters:
    -----------
    video_path : str
        Путь к видео файлу
    metadata_path : str
        Путь к JSON файлу с аннотациями
    output_dir : str
        Директория для сохранения кадров и аннотаций
    target_width : int
        Целевая ширина кадров (по умолчанию 1280)
    target_height : int
        Целевая высота кадров (по умолчанию 960)
    """
    # Загрузка метаданных
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    annotations_list = metadata.get('annotations', [])
    
    # Создание словаря для быстрого доступа к аннотациям по frame_id
    annotations_dict = {}
    for ann in annotations_list:
        frame_id = ann.get('frame_id')
        annotations_dict[frame_id] = ann.get('annotations', [])
    
    # Открытие видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    # Получение оригинальных размеров видео
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Вычисление коэффициентов масштабирования
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height
    
    frame_id = 0
    extracted_frames = []
    
    # Создание директорий
    images_dir = Path(output_dir) / 'images'
    labels_dir = Path(output_dir) / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Извлечение кадров из видео: {video_path}")
    print(f"Оригинальный размер: {orig_width}x{orig_height}")
    print(f"Целевой размер: {target_width}x{target_height}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Пропускаем кадры без аннотаций
        if frame_id not in annotations_dict:
            frame_id += 1
            continue
        
        # Изменение размера кадра
        resized_frame = cv2.resize(frame, (target_width, target_height))
        
        # Сохранение кадра
        frame_filename = f"frame_{frame_id:06d}.jpg"
        frame_path = images_dir / frame_filename
        cv2.imwrite(str(frame_path), resized_frame)
        
        # Конвертация аннотаций в формат YOLO
        yolo_annotations = []
        for obj_ann in annotations_dict[frame_id]:
            bbox = obj_ann.get('bbox')
            if not bbox or len(bbox) != 2:
                continue
            
            # Масштабирование координат bbox
            scaled_bbox = [
                [bbox[0][0] * scale_x, bbox[0][1] * scale_y],
                [bbox[1][0] * scale_x, bbox[1][1] * scale_y]
            ]
            
            # Конвертация в формат YOLO
            x_center, y_center, width, height = convert_bbox_to_yolo(
                scaled_bbox, target_width, target_height
            )
            
            # Получение class_id (обычно 0 для мыши)
            class_id = obj_ann.get('category_id', 0)
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Сохранение аннотаций
        label_filename = f"frame_{frame_id:06d}.txt"
        label_path = labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        extracted_frames.append({
            'frame_id': frame_id,
            'image_path': str(frame_path),
            'label_path': str(label_path)
        })
        
        frame_id += 1
        
        if frame_id % 100 == 0:
            print(f"Обработано кадров: {frame_id}")
    
    cap.release()
    print(f"Всего извлечено кадров с аннотациями: {len(extracted_frames)}")
    
    return extracted_frames


def split_dataset(frames_data, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Разделяет датасет на train/val/test
    
    Parameters:
    -----------
    frames_data : list
        Список словарей с информацией о кадрах
    output_dir : str
        Базовая директория для выходных данных
    train_ratio : float
        Доля тренировочных данных (по умолчанию 0.7)
    val_ratio : float
        Доля валидационных данных (по умолчанию 0.2)
    test_ratio : float
        Доля тестовых данных (по умолчанию 0.1)
    """
    import shutil
    
    # Разделение на train и temp (val + test)
    train_data, temp_data = train_test_split(
        frames_data, test_size=(val_ratio + test_ratio), random_state=42
    )
    
    # Разделение temp на val и test
    val_data, test_data = train_test_split(
        temp_data, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42
    )
    
    # Создание структуры директорий
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        split_images_dir = Path(output_dir) / 'images' / split_name
        split_labels_dir = Path(output_dir) / 'labels' / split_name
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Копирование файлов
        for frame_info in split_data:
            image_path = Path(frame_info['image_path'])
            label_path = Path(frame_info['label_path'])
            
            # Копирование изображения
            new_image_path = split_images_dir / image_path.name
            shutil.copy2(image_path, new_image_path)
            
            # Копирование аннотации
            new_label_path = split_labels_dir / label_path.name
            shutil.copy2(label_path, new_label_path)
    
    # Удаление исходных директорий после копирования
    images_dir = Path(output_dir) / 'images'
    labels_dir = Path(output_dir) / 'labels'
    for file in images_dir.glob('*.jpg'):
        file.unlink()
    for file in labels_dir.glob('*.txt'):
        file.unlink()
    
    print(f"Разделение датасета:")
    print(f"  Train: {len(train_data)} кадров")
    print(f"  Val: {len(val_data)} кадров")
    print(f"  Test: {len(test_data)} кадров")
    
    return len(train_data), len(val_data), len(test_data)


if __name__ == "__main__":
    # Пути к данным
    video_path = "tracking_datasets_extracted/tracking_datasets/single_animal_tracking/epm_1/epm_1.mp4"
    metadata_path = "tracking_datasets_extracted/tracking_datasets/single_animal_tracking/epm_1/metadata.json"
    output_dir = "yolo_dataset_epm1"
    
    # Размеры кадров
    target_width = 1280
    target_height = 960
    
    # Извлечение кадров и создание аннотаций
    frames_data = extract_frames_and_annotations(
        video_path, metadata_path, output_dir, target_width, target_height
    )
    
    # Разделение на train/val/test
    split_dataset(frames_data, output_dir)

