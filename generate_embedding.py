import os
import json
import cv2
import numpy as np
from deepface import DeepFace
# from deepface.commons import functions
from mtcnn import MTCNN
import warnings
warnings.filterwarnings('ignore')

class FaceEmbeddingGenerator:
    def __init__(self, config=None):
        """Initialize the embedding generator with configuration"""
        self.config = config or self.get_default_config()
        self.detector = MTCNN() if self.config['face_alignment']['use_mtcnn'] else None
        
    def get_default_config(self):
        """Default configuration for embedding generation"""
        return {
            'image_resize': {
                'enabled': True,
                'target_size': (224, 224),
                'maintain_aspect_ratio': True
            },
            'augmentation': {
                'enabled': True,
                'brightness_variation': {
                    'enabled': True,
                    'min_factor': 0.7,
                    'max_factor': 1.3,
                    'num_variations': 3
                },
                'lighting_variation': {
                    'enabled': True,
                    'positions': ['left', 'right', 'top', 'bottom', 'center'],
                    'intensity_range': (0.3, 0.8),
                    'decay_factor': 0.5
                }
            },
            'face_alignment': {
                'use_mtcnn': True,
                'backends': ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
            },
            'face_recognition': {
                'models': ['Facenet', 'ArcFace', 'GhostFaceNet'],
                'normalization': 'base'
            }
        }
    
    def resize_image(self, image):
        """Resize image based on configuration"""
        if not self.config['image_resize']['enabled']:
            return image
            
        target_size = self.config['image_resize']['target_size']
        
        if self.config['image_resize']['maintain_aspect_ratio']:
            h, w = image.shape[:2]
            aspect = w / h
            
            if aspect > 1:
                new_w = target_size[0]
                new_h = int(new_w / aspect)
            else:
                new_h = target_size[1]
                new_w = int(new_h * aspect)
                
            resized = cv2.resize(image, (new_w, new_h))
            
            # Pad to target size
            pad_w = target_size[0] - new_w
            pad_h = target_size[1] - new_h
            
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            
            resized = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            resized = cv2.resize(image, target_size)
            
        return resized
    
    def apply_brightness_variation(self, image, factor):
        """Apply brightness variation to image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def apply_lighting_variation(self, image, position, intensity):
        """Apply spherical lighting effect"""
        h, w = image.shape[:2]
        
        # Define light position
        positions = {
            'left': (0, h // 2),
            'right': (w, h // 2),
            'top': (w // 2, 0),
            'bottom': (w // 2, h),
            'center': (w // 2, h // 2)
        }
        
        light_x, light_y = positions[position]
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Calculate distance from light source
        distance = np.sqrt((x - light_x) ** 2 + (y - light_y) ** 2)
        max_distance = np.sqrt(w ** 2 + h ** 2)
        
        # Create lighting mask with decay
        decay_factor = self.config['augmentation']['lighting_variation']['decay_factor']
        lighting_mask = intensity * np.exp(-decay_factor * distance / max_distance)
        
        # Apply lighting effect
        result = image.astype(np.float32)
        for i in range(3):
            result[:, :, i] = result[:, :, i] + lighting_mask * 255
            
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    def augment_image(self, image):
        """Generate augmented versions of the image"""
        augmented_images = [image]  # Include original
        
        if not self.config['augmentation']['enabled']:
            return augmented_images
        
        # Brightness variations
        if self.config['augmentation']['brightness_variation']['enabled']:
            brightness_config = self.config['augmentation']['brightness_variation']
            factors = np.linspace(
                brightness_config['min_factor'],
                brightness_config['max_factor'],
                brightness_config['num_variations']
            )
            
            for factor in factors:
                if factor != 1.0:  # Skip original brightness
                    augmented = self.apply_brightness_variation(image, factor)
                    augmented_images.append(augmented)
        
        # Lighting variations
        if self.config['augmentation']['lighting_variation']['enabled']:
            lighting_config = self.config['augmentation']['lighting_variation']
            
            for position in lighting_config['positions']:
                intensities = np.linspace(
                    lighting_config['intensity_range'][0],
                    lighting_config['intensity_range'][1],
                    lighting_config['num_variations']
                )
                for intensity in intensities:
                    augmented = self.apply_lighting_variation(image, position, intensity)
                    augmented_images.append(augmented)
        
        return augmented_images
    
    def align_face(self, image, backend='opencv'):
        """Align face using specified backend"""
        try:
            if backend == 'mtcnn' and self.detector:
                # Use MTCNN for face detection
                faces = self.detector.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if faces:
                    x, y, w, h = faces[0]['box']
                    face = image[y:y+h, x:x+w]
                    return cv2.resize(face, (160, 160))
            
            # Use DeepFace's built-in alignment
            face_objs = DeepFace.extract_faces(
                img_path=image,
                detector_backend=backend,
                enforce_detection=False,
                align=True
            )
            
            if face_objs:
                return (face_objs[0]['face'] * 255).astype(np.uint8)
                
        except Exception as e:
            print(f"Face alignment failed with {backend}: {str(e)}")
            
        return image
    
    def generate_embeddings(self, image, person_name, image_name):
        """Generate embeddings for all models and augmentations"""
        embeddings = {
            'person': person_name,
            'original_image': image_name,
            'embeddings': {}
        }
        
        # Get augmented images
        augmented_images = self.augment_image(image)
        
        # Process each model
        for model_name in self.config['face_recognition']['models']:
            embeddings['embeddings'][model_name] = []
            
            # Process each augmented image
            for aug_idx, aug_image in enumerate(augmented_images):
                # Try different alignment backends
                for backend in self.config['face_alignment']['backends']:
                    try:
                        # Align face
                        aligned_face = self.align_face(aug_image, backend)
                        
                        # Generate embedding
                        embedding = DeepFace.represent(
                            img_path=aligned_face,
                            model_name=model_name,
                            enforce_detection=False,
                            detector_backend=backend,
                            normalization=self.config['face_recognition']['normalization']
                        )
                        
                        if embedding:
                            embeddings['embeddings'][model_name].append({
                                'augmentation_idx': aug_idx,
                                'alignment_backend': backend,
                                'embedding': embedding[0]['embedding']
                            })
                            
                    except Exception as e:
                        print(f"Failed to generate embedding with {model_name} and {backend}: {str(e)}")
                        continue
        
        return embeddings
    
    def process_person_folder(self, person_folder, person_name):
        """Process all images in a person's folder"""
        person_embeddings = []
        
        for image_file in os.listdir(person_folder):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(person_folder, image_file)
                print(f"Processing {person_name}/{image_file}")
                
                # Load and resize image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                    
                image = self.resize_image(image)
                
                # Generate embeddings
                embeddings = self.generate_embeddings(image, person_name, image_file)
                person_embeddings.append(embeddings)
        
        return person_embeddings
    
    def generate_database(self, input_folder, output_file='embeddings_db.json'):
        """Generate embedding database from labeled folders"""
        database = {
            'config': self.config,
            'persons': {}
        }
        
        # Process each person folder
        for person_name in os.listdir(input_folder):
            person_folder = os.path.join(input_folder, person_name)
            
            if os.path.isdir(person_folder):
                print(f"\nProcessing person: {person_name}")
                person_embeddings = self.process_person_folder(person_folder, person_name)
                database['persons'][person_name] = person_embeddings
        
        # Save database
        with open(output_file, 'w') as f:
            json.dump(database, f, indent=2)
        
        print(f"\nEmbedding database saved to {output_file}")
        return database


def main():
    print("Starting Face Embedding Generation...")
    # Example configuration
    config = {
        'image_resize': {
            'enabled': True,
            'target_size': (224, 224),
            'maintain_aspect_ratio': True
        },
        'augmentation': {
            'enabled': True,
            'brightness_variation': {
                'enabled': True,
                'min_factor': 0.7,
                'max_factor': 1.3,
                'num_variations': 6
            },
            'lighting_variation': {
                'enabled': True,
                'positions': ['left', 'right', 'top', 'bottom'],
                'intensity_range': (0.3, 0.8),
                'num_variations': 10,
                'decay_factor': 0.5
            }
        },
        'face_alignment': {
            'use_mtcnn': False,
            # 'backends': ['opencv', 'mtcnn', 'retinaface']
            'backends': ['mtcnn']
        },
        'face_recognition': {
            'models': ['Facenet', 'ArcFace', 'GhostFaceNet'],
            'normalization': 'base'
        }
    }
    
    # Initialize generator
    generator = FaceEmbeddingGenerator(config)
    
    # Generate database
    input_folder = 'labeled_faces'  # Folder structure: labeled_faces/person_name/images
    output_file = 'embeddings_db.json'
    
    generator.generate_database(input_folder, output_file)


if __name__ == "__main__":
    main()