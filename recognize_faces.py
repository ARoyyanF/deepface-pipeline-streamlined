import os
import json
import cv2
import numpy as np
import shutil
from deepface import DeepFace
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FaceRecognizer:
    def __init__(self, database_path, config=None):
        """Initialize face recognizer with database and configuration"""
        self.database = self.load_database(database_path)
        self.config = config or self.get_default_config()
        self.results = []
        
    def get_default_config(self):
        """Default configuration for face recognition"""
        return {
            'models': [
                {
                    'name': 'Facenet',
                    'alignment_backend': 'opencv',
                    'distance_metric': 'cosine',  # cosine, euclidean, euclidean_l2
                    'threshold': 0.40,
                    'normalization': 'base'
                },
                {
                    'name': 'ArcFace',
                    'alignment_backend': 'mtcnn',
                    'distance_metric': 'cosine',
                    'threshold': 0.68,
                    'normalization': 'base'
                },
                {
                    'name': 'GhostFaceNet',
                    'alignment_backend': 'retinaface',
                    'distance_metric': 'cosine',
                    'threshold': 0.65,
                    'normalization': 'base'
                }
            ],
            'fallback_conditions': {
                'max_matches_different_persons': 2,
                'min_distance_difference': 0.1,
                'use_threshold': True
            },
            'image_resize': {
                'enabled': True,
                'target_size': (224, 224)
            },
            'output': {
                'results_file': 'results.json',
                'output_folder': 'recognized_faces',
                'save_failed': True,
                'failed_folder': 'failed_recognition'
            }
        }
    
    def load_database(self, database_path):
        """Load embedding database from JSON file"""
        with open(database_path, 'r') as f:
            return json.load(f)
        
    def calculate_distance(self, embedding1, embedding2, metric='cosine'):
        """Calculate distance between two embeddings using mathematical formulas."""
        # Ensure embeddings are numpy arrays for calculations
        embedding1 = np.asarray(embedding1)
        embedding2 = np.asarray(embedding2)

        if metric == 'cosine':
            # Cosine distance is 1 - cosine similarity
            # Cosine Similarity = (A · B) / (||A|| * ||B||)
            dot_product = np.dot(embedding1, embedding2)
            norm_1 = np.linalg.norm(embedding1)
            norm_2 = np.linalg.norm(embedding2)
            
            # Add a small value (epsilon) to avoid division by zero
            similarity = dot_product / ((norm_1 * norm_2) + 1e-6)
            return 1 - similarity

        elif metric == 'euclidean':
            # Euclidean distance is the L2 norm of the difference between the vectors
            # Formula: sqrt(Σ(aᵢ - bᵢ)²)
            return np.linalg.norm(embedding1 - embedding2)

        elif metric == 'euclidean_l2':
            # First, L2-normalize both vectors
            # A_normalized = A / ||A||
            norm_1 = np.linalg.norm(embedding1)
            norm_2 = np.linalg.norm(embedding2)
            normalized_emb1 = embedding1 / (norm_1 + 1e-6) # Add epsilon
            normalized_emb2 = embedding2 / (norm_2 + 1e-6) # Add epsilon
            
            # Then, calculate the Euclidean distance between the normalized vectors
            return np.linalg.norm(normalized_emb1 - normalized_emb2)

        else:
            raise ValueError(f"Unknown distance metric: {metric}")
    
    def resize_image(self, image):
        """Resize image if enabled in config"""
        if not self.config['image_resize']['enabled']:
            return image
            
        target_size = self.config['image_resize']['target_size']
        return cv2.resize(image, target_size)
    
    def generate_embedding(self, image, model_name, alignment_backend, normalization='base'):
        """Generate embedding for a single image"""
        try:
            embedding = DeepFace.represent(
                img_path=image,
                model_name=model_name,
                enforce_detection=False,
                detector_backend=alignment_backend,
                normalization=normalization
            )
            
            if embedding:
                return embedding[0]['embedding']
                
        except Exception as e:
            print(f"Failed to generate embedding with {model_name}: {str(e)}")
            
        return None
    
    def find_matches(self, test_embedding, model_config):
        """Find matches in database for given embedding"""
        matches = []
        model_name = model_config['name']
        distance_metric = model_config['distance_metric']
        
        # Search through all persons in database
        for person_name, person_data in self.database['persons'].items():
            for image_data in person_data:
                model_embeddings = image_data['embeddings'].get(model_name, [])
                
                for emb_data in model_embeddings:
                    db_embedding = emb_data['embedding']
                    distance = self.calculate_distance(
                        test_embedding, 
                        db_embedding, 
                        distance_metric
                    )
                    
                    matches.append({
                        'person': person_name,
                        'distance': distance,
                        'original_image': image_data['original_image'],
                        'alignment_backend': emb_data['alignment_backend'],
                        'augmentation_idx': emb_data['augmentation_idx']
                    })
        
        # Sort by distance
        matches.sort(key=lambda x: x['distance'])
        return matches
    
    def check_fallback_conditions(self, matches, threshold):
        """Check if fallback conditions are met"""
        fallback_config = self.config['fallback_conditions']
        
        # No matches found
        if not matches:
            return True, "No matches found"
        
        # Check threshold condition
        if fallback_config['use_threshold']:
            valid_matches = [m for m in matches if m['distance'] <= threshold]
            
            if not valid_matches:
                return True, f"No matches below threshold {threshold}"
            
            # Check multiple different persons condition
            unique_persons = set(m['person'] for m in valid_matches[:fallback_config['max_matches_different_persons']])
            if len(unique_persons) > 1:
                return True, f"Multiple persons ({len(unique_persons)}) matched above threshold"
            
            # Check distance difference with the next best match
        if len(valid_matches) > 1:
            best_match = valid_matches[0]
            second_best_match = valid_matches[1]
            distance_diff = second_best_match['distance'] - best_match['distance']
            
            # Trigger fallback only if the distance is too close AND it's a DIFFERENT person.
            if distance_diff < fallback_config['min_distance_difference']:
                if best_match['person'] != second_best_match['person']:
                    return True, (f"Ambiguous: distance diff ({distance_diff:.3f}) to a different person "
                                  f"is below minimum ({fallback_config['min_distance_difference']})")
        
        return False, "Conditions met"
    
    def recognize_face(self, image_path, image_name):
        """Recognize face using multi-model fallback approach"""
        result = {
            'image': image_name,
            'timestamp': datetime.now().isoformat(),
            'models_tried': [],
            'final_result': None,
            'success': False
        }
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            result['error'] = "Failed to load image"
            return result
            
        image = self.resize_image(image)
        
        # Try each model in sequence
        for model_config in self.config['models']:
            model_name = model_config['name']
            print(f"  Trying {model_name}...")
            
            model_result = {
                'model': model_name,
                'alignment_backend': model_config['alignment_backend'],
                'distance_metric': model_config['distance_metric'],
                'threshold': model_config['threshold']
            }
            
            # Generate embedding
            embedding = self.generate_embedding(
                image,
                model_name,
                model_config['alignment_backend'],
                model_config.get('normalization', 'base')
            )
            
            if embedding is None:
                model_result['error'] = "Failed to generate embedding"
                result['models_tried'].append(model_result)
                continue
            
            # Find matches
            matches = self.find_matches(embedding, model_config)
            model_result['total_matches'] = len(matches)
            
            if matches:
                model_result['best_match'] = {
                    'person': matches[0]['person'],
                    'distance': matches[0]['distance'],
                    'original_image': matches[0]['original_image']
                }
                
                # Add top 5 matches for debugging
                model_result['top_matches'] = [
                    {
                        'person': m['person'],
                        'distance': m['distance'],
                        'original_image': m['original_image'],
                        'alignment_backend': m['alignment_backend'],
                        'augmentation_idx': m['augmentation_idx']
                    } for m in matches[:5]
                ]
            
            # Check fallback conditions
            should_fallback, reason = self.check_fallback_conditions(
                matches, 
                model_config['threshold']
            )
            
            model_result['fallback_triggered'] = should_fallback
            model_result['fallback_reason'] = reason
            
            result['models_tried'].append(model_result)
            
            if not should_fallback and matches:
                # Success! Found a good match
                result['success'] = True
                result['final_result'] = {
                    'person': matches[0]['person'],
                    'distance': matches[0]['distance'],
                    'original_image': matches[0]['original_image'],
                    'model': model_name,
                    'confidence': 1.0 - (matches[0]['distance'] / model_config['threshold'])
                }
                break
        
        return result
    
    def save_result_image(self, image_path, result):
        """Save result image to appropriate folder"""
        output_config = self.config['output']
        
        if result['success']:
            # Create person folder
            person_folder = os.path.join(
                output_config['output_folder'],
                result['final_result']['person']
            )
            os.makedirs(person_folder, exist_ok=True)
            
            # Create filename with model and distance info
            model = result['final_result']['model']
            distance = result['final_result']['distance']
            original_image = result['final_result']['original_image']
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            ext = os.path.splitext(image_path)[1]
            
            output_filename = f"{base_name}_{model}_{original_image}_d{distance:.3f}{ext}"
            output_path = os.path.join(person_folder, output_filename)
            
        elif output_config['save_failed']:
            # Save failed recognition
            failed_folder = output_config['failed_folder']
            os.makedirs(failed_folder, exist_ok=True)
            
            base_name = os.path.basename(image_path)
            output_path = os.path.join(failed_folder, base_name)
        else:
            return
            
        # Copy image
        shutil.copy2(image_path, output_path)
    
    def process_folder(self, input_folder):
        """Process all images in input folder"""
        print(f"Processing images from {input_folder}")
        
        # Create output directories
        os.makedirs(self.config['output']['output_folder'], exist_ok=True)
        if self.config['output']['save_failed']:
            os.makedirs(self.config['output']['failed_folder'], exist_ok=True)
        
        # Process each image
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for idx, image_file in enumerate(image_files):
            print(f"\nProcessing {idx+1}/{len(image_files)}: {image_file}")
            image_path = os.path.join(input_folder, image_file)
            
            # Recognize face
            result = self.recognize_face(image_path, image_file)
            self.results.append(result)
            
            # Save result image
            self.save_result_image(image_path, result)
            
            # Print summary
            if result['success']:
                print(f"  ✓ Recognized: {result['final_result']['person']} "
                      f"(model: {result['final_result']['model']}, "
                      f"distance: {result['final_result']['distance']:.3f})")
            else:
                print(f"  ✗ Failed to recognize")
        
        # Save results
        self.save_results()
        
    def save_results(self):
        """Save recognition results to JSON file"""
        output_file = self.config['output']['results_file']
        
        summary = {
            'total_images': len(self.results),
            'successful': sum(1 for r in self.results if r['success']),
            'failed': sum(1 for r in self.results if not r['success']),
            'config': self.config,
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        print(f"Summary: {summary['successful']}/{summary['total_images']} recognized successfully")


def main():
    # Example configuration
    config = {
        'models': [
            {
                'name': 'Facenet',
                'alignment_backend': 'mtcnn',
                'distance_metric': 'cosine',
                'threshold': 0.40,
                'normalization': 'base'
            },
            # {
            #     'name': 'ArcFace',
            #     'alignment_backend': 'mtcnn',
            #     'distance_metric': 'cosine',
            #     'threshold': 0.68,
            #     'normalization': 'base'
            # },
            {
                'name': 'GhostFaceNet',
                'alignment_backend': 'mtcnn',
                'distance_metric': 'cosine',
                'threshold': 0.65,
                'normalization': 'base'
            }
        ],
        'fallback_conditions': {
            'max_matches_different_persons': 1,
            'min_distance_difference': 0.1,
            'use_threshold': True
        },
        'image_resize': {
            'enabled': True,
            'target_size': (224, 224)
        },
        'output': {
            'results_file': 'results.json',
            'output_folder': 'recognized_faces',
            'save_failed': True,
            'failed_folder': 'failed_recognition'
        }
    }
    
    # Initialize recognizer
    database_path = 'embeddings_db.json'
    recognizer = FaceRecognizer(database_path, config)
    
    # Process test images
    input_folder = 'test_images'
    recognizer.process_folder(input_folder)


if __name__ == "__main__":
    main()