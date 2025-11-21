"""
Raspberry Pi Coffee Disease Detection Monitor - UPDATED
Uses direct pi-save.php and pi-alert.php endpoints
"""

import os
import sys
import time
import json
import logging
import requests
from datetime import datetime
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Try to import Pi Camera
try:
    from picamera2 import Picamera2
    PI_CAMERA_AVAILABLE = True
except ImportError:
    PI_CAMERA_AVAILABLE = False
    print("Warning: picamera2 not available. Using test mode.")

# Configuration
CONFIG_FILE = "config.json"
LOG_FILE = "disease_monitor.log"
IMAGE_SAVE_DIR = "captured_images"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CoffeeDiseaseMonitor:
    def __init__(self, config_path=CONFIG_FILE):
        """Initialize the disease monitor"""
        self.config = self.load_config(config_path)
        self.model = None
        self.camera = None
        self.session = requests.Session()
        self.auth_token = None
        
        # Disease classes
        self.CLASS_NAMES = ['BROWN EYE SPOT', 'CBD', 'HEALTHY', 'LEAF RUST', 'SOOTY MOULD']
        self.CONFIDENCE_THRESHOLD = 0.6
        
        # Disease treatment information
        self.DISEASE_TREATMENTS = {
            "CBD": {
                "full_name": "Coffee Berry Disease",
                "severity": "High",
                "immediate_actions": [
                    "Remove and destroy all infected berries and leaves immediately",
                    "Prune affected branches at least 30cm below visible infection",
                    "Burn or bury infected plant material away from the plantation"
                ],
                "fungicide_treatment": {
                    "fungicides": [
                        "Copper-based fungicides (Copper oxychloride at 3-4g/L)",
                        "Chlorothalonil (2-3ml/L)",
                        "Azoxystrobin + Difenoconazole combination"
                    ],
                    "schedule": "Spray every 2-3 weeks during wet season, starting at flowering",
                    "method": "Thorough coverage of berries, flowers, and leaves"
                },
                "cultural_practices": [
                    "Improve air circulation by proper spacing and pruning",
                    "Avoid overhead irrigation during flowering and fruiting",
                    "Maintain proper nutrition (avoid excess nitrogen)",
                    "Mulch to reduce soil splash onto lower leaves"
                ],
                "preventive_measures": [
                    "Use resistant varieties like Ruiru 11 or Batian",
                    "Regular monitoring especially during rainy season",
                    "Maintain plantation hygiene"
                ],
                "cost": "Ksh 6000-10000 per acre per season"
            },
            "LEAF RUST": {
                "full_name": "Coffee Leaf Rust (Hemileia vastatrix)",
                "severity": "High",
                "immediate_actions": [
                    "Remove severely infected leaves to reduce spore load",
                    "Increase shade management to reduce leaf wetness",
                    "Apply fungicide treatment immediately"
                ],
                "fungicide_treatment": {
                    "fungicides": [
                        "Copper-based fungicides (2-3g/L)",
                        "Triazole fungicides (Cyproconazole, Tebuconazole at 1-2ml/L)",
                        "Systemic: Azoxystrobin or Pyraclostrobin"
                    ],
                    "schedule": "4-6 applications per year, starting before rainy season",
                    "method": "Spray undersides of leaves thoroughly, early morning application preferred"
                },
                "cultural_practices": [
                    "Optimize shade levels (30-40% shade)",
                    "Maintain adequate plant nutrition with balanced NPK",
                    "Proper weed control to improve air circulation",
                    "Avoid water stress"
                ],
                "preventive_measures": [
                    "Plant rust-resistant varieties (Ruiru 11, Batian)",
                    "Monitor regularly (weekly during wet season)",
                    "Maintain proper spacing (2.5m x 2.5m or wider)"
                ],
                "cost": "Ksh 5000-9000 per acre per season"
            },
            "BROWN EYE SPOT": {
                "full_name": "Brown Eye Spot (Cercospora coffeicola)",
                "severity": "Medium",
                "immediate_actions": [
                    "Remove heavily infected leaves",
                    "Improve plantation drainage",
                    "Reduce plant stress factors"
                ],
                "fungicide_treatment": {
                    "fungicides": [
                        "Copper-based fungicides (Copper hydroxide at 3g/L)",
                        "Mancozeb (2-3g/L)",
                        "Chlorothalonil (2ml/L)"
                    ],
                    "schedule": "2-3 applications at 3-week intervals during rainy season",
                    "method": "Spray both leaf surfaces, especially lower canopy"
                },
                "cultural_practices": [
                    "Improve soil nutrition (especially potassium)",
                    "Manage shade to reduce humidity",
                    "Remove weeds that harbor the pathogen",
                    "Ensure proper drainage to avoid waterlogging"
                ],
                "preventive_measures": [
                    "Maintain optimal shade (not too dense)",
                    "Regular fertilization with emphasis on potassium",
                    "Avoid overcrowding of plants"
                ],
                "cost": "Ksh 4000-8000 per acre per season"
            },
            "SOOTY MOULD": {
                "full_name": "Sooty Mould (Secondary infection from scale insects)",
                "severity": "Medium",
                "immediate_actions": [
                    "Identify and control the primary pest (scale insects, mealybugs, or aphids)",
                    "Wash off sooty mould with water spray",
                    "Prune heavily affected branches"
                ],
                "fungicide_treatment": {
                    "fungicides": [
                        "Copper-based fungicides (mild solution: 2g/L)",
                        "Note: Fungicides are secondary - focus on pest control"
                    ],
                    "schedule": "Apply after pest control measures",
                    "method": "Light spray on affected areas"
                },
                "insecticide_treatment": {
                    "insecticides": [
                        "Horticultural oil spray (2-3%)",
                        "Imidacloprid for systemic control",
                        "Insecticidal soap solution",
                        "Neem oil (organic option)"
                    ],
                    "schedule": "2-3 applications at 10-14 day intervals"
                },
                "cultural_practices": [
                    "Encourage natural predators (ladybugs, lacewings)",
                    "Prune to improve air circulation",
                    "Remove ant colonies (ants protect scale insects)",
                    "Wash leaves with water to remove honeydew"
                ],
                "preventive_measures": [
                    "Regular monitoring for pest infestations",
                    "Maintain beneficial insect populations",
                    "Avoid excessive nitrogen fertilization"
                ],
                "cost": "Ksh 3000-6000 per acre per season"
            },
            "HEALTHY": {
                "full_name": "Healthy Coffee Plant",
                "severity": "None",
                "immediate_actions": [
                    "Continue current management practices",
                    "Maintain regular monitoring schedule"
                ],
                "preventive_measures": [
                    "Apply preventive fungicide spray during high-risk periods",
                    "Maintain balanced nutrition (NPK + micronutrients)",
                    "Ensure proper pruning and plant spacing",
                    "Regular scouting for early disease detection",
                    "Maintain mulch layer around plants",
                    "Proper shade management (30-40%)",
                    "Good drainage and irrigation management"
                ],
                "cultural_practices": [
                    "Nitrogen: 100-150 kg/ha/year (split applications)",
                    "Phosphorus: 40-60 kg/ha/year",
                    "Potassium: 120-180 kg/ha/year",
                    "Weekly visual inspection during rainy season"
                ],
                "cost": "Ksh 15000-20000 per acre per season (maintenance)"
            }
        }
        
        # Create image save directory
        Path(IMAGE_SAVE_DIR).mkdir(exist_ok=True)
        
        logger.info("Coffee Disease Monitor initialized")
    
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file {config_path} not found!")
            self.create_sample_config(config_path)
            sys.exit(1)
    
        try:
            with open(config_path, 'w') as f:
                json.dump(sample_config, f, indent=2)
            logger.info(f"Sample configuration created at {config_path}")
            logger.info("Please edit the configuration file with your actual values")
        except Exception as e:
            logger.error(f"Failed to create sample config: {e}")

    def load_model(self):
        """Load the TensorFlow disease detection model"""
        try:
            model_path = self.config['detection']['model_path']
            logger.info(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize the camera"""
        if not PI_CAMERA_AVAILABLE:
            logger.warning("Pi Camera not available, running in test mode")
            return False
        
        try:
            self.camera = Picamera2()
            config = self.camera.create_still_configuration(
                main={"size": tuple(self.config['camera']['resolution'])}
            )
            self.camera.configure(config)
            self.camera.start()
            
            warmup = self.config['camera']['warmup_time']
            logger.info(f"Camera warming up for {warmup} seconds...")
            time.sleep(warmup)
            
            logger.info("Camera initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def authenticate(self):
        """Authenticate with the API and get token"""
        try:
            login_url = f"{self.config['api']['base_url']}/auth/login.php"
            
            payload = {
                "email": self.config['farmer']['email'],
                "password": self.config['farmer']['password']
            }
            
            logger.info(f"Authenticating as {payload['email']}")
            
            response = self.session.post(
                login_url,
                json=payload,
                verify=self.config['api']['verify_ssl'],
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    self.auth_token = data['data']['token']
                    user = data['data']['user']
                    logger.info(f"✅ Authentication successful for {user['full_name']}")
                    
                    # Update session headers
                    self.session.headers.update({
                        'Authorization': f'Bearer {self.auth_token}'
                    })
                    
                    return True
                else:
                    logger.error(f"❌ Authentication failed: {data.get('message')}")
                    return False
            else:
                logger.error(f"❌ Authentication failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            return False
    
    def capture_image(self):
        """Capture an image from the camera"""
        if PI_CAMERA_AVAILABLE and self.camera:
            try:
                logger.info("Capturing image with Pi Camera...")
                image_array = self.camera.capture_array()
                img = Image.fromarray(image_array)
                logger.info(f"Image captured: {img.size}")
                return img
            except Exception as e:
                logger.error(f"Pi Camera capture failed: {e}")
                logger.info("Falling back to USB camera...")
        
        # USB webcam fallback
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("No USB camera found")
                return None

            ret, frame = cap.read()
            cap.release()

            if not ret:
                logger.error("Failed to read frame from USB camera")
                return None

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            logger.info(f"Image captured with USB camera: {img.size}")
            return img

        except Exception as e:
            logger.error(f"Failed to capture image: {e}")
            return None

    def detect_disease(self, image):
        """Run disease detection on the image"""
        try:
            # Resize image
            img_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to array and normalize
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            logger.info("Running disease detection...")
            predictions = self.model.predict(img_array, verbose=0)
            
            confidence = float(np.max(predictions))
            predicted_class_idx = int(np.argmax(predictions))
            predicted_class = self.CLASS_NAMES[predicted_class_idx]
            
            logger.info(f"Detection result: {predicted_class} ({confidence*100:.2f}%)")
            
            # Check confidence threshold
            if confidence < self.CONFIDENCE_THRESHOLD:
                predicted_class = "UNCERTAIN"
                treatment = None
                logger.warning(f"Low confidence detection: {confidence*100:.2f}%")
            else:
                # Get treatment information
                treatment = self.DISEASE_TREATMENTS.get(predicted_class, None)
            
            return {
                'predicted_class': predicted_class,
                'confidence_score': confidence * 100,
                'treatment': treatment,
                'all_predictions': {
                    self.CLASS_NAMES[i]: float(predictions[0][i]) * 100 
                    for i in range(len(self.CLASS_NAMES))
                }
            }
            
        except Exception as e:
            logger.error(f"Disease detection failed: {e}")
            return None
    
    def save_detection_directly(self, image, detection_result):
        """
        Upload and save detection in one call using pi-save.php
        This replaces the old two-step process (detect.php + save.php)
        """
        try:
            # Save image temporarily for upload
            temp_image_path = os.path.join(IMAGE_SAVE_DIR, 'temp_upload.jpg')
            image.save(temp_image_path, 'JPEG', quality=95)
            
            logger.info(f"📤 Uploading detection to pi-save.php...")
            
            # Prepare the request
            upload_url = f"{self.config['api']['base_url']}/detection/pi-save.php"
            
            # Open file and prepare multipart form data
            with open(temp_image_path, 'rb') as img_file:
                files = {
                    'image': ('detection.jpg', img_file, 'image/jpeg')
                }
                
                # Include all metadata as form data
                data = {
                    'farm_id': self.config['farm']['farm_id'],
                    'farm_name': self.config['farm']['farm_name'],
                    'notes': f"Automated detection at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }
                
                # Add optional location data if available
                if 'location_lat' in self.config['farm']:
                    data['location_lat'] = self.config['farm']['location_lat']
                if 'location_lng' in self.config['farm']:
                    data['location_lng'] = self.config['farm']['location_lng']
                
                # Add treatment data if available
                if detection_result.get('treatment'):
                    data['treatment'] = json.dumps(detection_result['treatment'])
                
                headers = {
                    'Authorization': f'Bearer {self.auth_token}'
                }
                
                # Make request
                response = requests.post(
                    upload_url,
                    files=files,
                    data=data,
                    headers=headers,
                    verify=self.config['api']['verify_ssl'],
                    timeout=60
                )
            
            # Clean up temp file
            try:
                os.remove(temp_image_path)
            except:
                pass
            
            logger.info(f"Server response status: {response.status_code}")
            
            if response.status_code in [200, 201]:
                data = response.json()
                logger.info(f"Server response: {json.dumps(data, indent=2)}")
                
                if data.get('success'):
                    detection_id = data['data'].get('detection_id')
                    
                    if detection_id:
                        logger.info("=" * 60)
                        logger.info(f"✅ DETECTION SAVED SUCCESSFULLY!")
                        logger.info(f"   Detection ID: {detection_id}")
                        logger.info(f"   Disease: {data['data']['predicted_class']}")
                        logger.info(f"   Confidence: {data['data']['confidence_score']:.1f}%")
                        logger.info(f"   Status: Permanent (is_temporary=false)")
                        logger.info("=" * 60)
                        return data['data']
                    else:
                        logger.error("❌ Detection ID is null - save failed!")
                        return None
                else:
                    logger.error(f"❌ Upload failed: {data.get('message')}")
                    return None
            else:
                logger.error(f"❌ Upload failed with status {response.status_code}")
                try:
                    error_data = response.json()
                    logger.error(f"Error: {error_data}")
                except:
                    logger.error(f"Response: {response.text[:500]}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Save detection error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def send_alert_email(self, detection_data):
        """
        Send alert email using pi-alert.php
        Only called if disease is detected (not HEALTHY)
        """
        try:
            # Skip if healthy
            if detection_data['predicted_class'].upper() == 'HEALTHY':
                logger.info("✅ Plant is healthy - no alert needed")
                return True
            
            # Skip if notifications disabled
            if not self.config['notifications']['send_email_on_disease']:
                logger.info("📧 Email notifications disabled in config")
                return True
            
            alert_url = f"{self.config['api']['base_url']}/detection/pi-alert.php"
            
            # Use farmer's email as recipient
            recipient_email = self.config['farmer']['email']
            
            payload = {
                'detection_id': detection_data['detection_id'],
                'recipient_email': recipient_email
            }
            
            logger.info(f"📧 Sending alert email to {payload['recipient_email']}...")
            
            response = self.session.post(
                alert_url,
                json=payload,
                verify=self.config['api']['verify_ssl'],
                timeout=30
            )
            
            logger.info(f"Alert response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    logger.info("✅ Alert email sent successfully")
                    return True
                else:
                    logger.error(f"❌ Alert failed: {data.get('message')}")
                    return False
            else:
                logger.error(f"❌ Alert failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Alert email error: {e}")
            return False
    
    def save_image_locally(self, image, detection_result):
        """Save image to local storage"""
        try:
            # Only save if disease detected (if configured)
            if self.config['detection']['only_save_diseases']:
                if detection_result['predicted_class'] == 'HEALTHY':
                    return
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            disease = detection_result['predicted_class'].replace(' ', '_')
            confidence = detection_result['confidence_score']
            
            filename = f"{timestamp}_{disease}_{confidence:.1f}.jpg"
            filepath = os.path.join(IMAGE_SAVE_DIR, filename)
            
            image.save(filepath, 'JPEG', quality=95)
            logger.info(f"💾 Image saved locally: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save image locally: {e}")
    
    def display_treatment_info(self, detection_result):
        """Display treatment information for detected disease"""
        try:
            predicted_class = detection_result['predicted_class']
            
            # Skip if healthy or uncertain
            if predicted_class in ['HEALTHY', 'UNCERTAIN']:
                return
            
            treatment = detection_result.get('treatment')
            if not treatment:
                return
            
            logger.info("=" * 60)
            logger.info(f"🔬 DISEASE TREATMENT INFORMATION")
            logger.info("=" * 60)
            logger.info(f"Disease: {treatment['full_name']}")
            logger.info(f"Severity: {treatment['severity']}")
            logger.info("")
            
            logger.info("⚠️  IMMEDIATE ACTIONS:")
            for i, action in enumerate(treatment['immediate_actions'], 1):
                logger.info(f"  {i}. {action}")
            logger.info("")
            
            if 'fungicide_treatment' in treatment:
                fung = treatment['fungicide_treatment']
                logger.info("💊 FUNGICIDE TREATMENT:")
                logger.info(f"  Recommended fungicides:")
                for fungicide in fung['fungicides']:
                    logger.info(f"    • {fungicide}")
                logger.info(f"  Schedule: {fung['schedule']}")
                logger.info(f"  Method: {fung['method']}")
                logger.info("")
            
            if 'insecticide_treatment' in treatment:
                insect = treatment['insecticide_treatment']
                logger.info("🐛 INSECTICIDE TREATMENT:")
                logger.info(f"  Recommended insecticides:")
                for insecticide in insect['insecticides']:
                    logger.info(f"    • {insecticide}")
                logger.info(f"  Schedule: {insect['schedule']}")
                logger.info("")
            
            logger.info("🌱 CULTURAL PRACTICES:")
            for i, practice in enumerate(treatment['cultural_practices'], 1):
                logger.info(f"  {i}. {practice}")
            logger.info("")
            
            logger.info(f"💰 Estimated Cost: {treatment['cost']}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to display treatment info: {e}")
    
    def run_detection_cycle(self):
        """Run one complete detection cycle"""
        logger.info("=" * 60)
        logger.info("Starting detection cycle")
        
        # Capture image
        image = self.capture_image()
        if image is None:
            logger.error("Failed to capture image, skipping cycle")
            return False
        
        # Run detection
        detection_result = self.detect_disease(image)
        if detection_result is None:
            logger.error("Failed to detect disease, skipping cycle")
            return False
        
        # Display treatment information if disease detected
        self.display_treatment_info(detection_result)
        
        # Save detection directly using pi-save.php
        saved_data = self.save_detection_directly(image, detection_result)
        if saved_data is None:
            logger.error("❌ Failed to save detection to server")
            # Still save locally
            self.save_image_locally(image, detection_result)
            return False
        
        # Send alert email if disease detected
        if saved_data['predicted_class'].upper() != 'HEALTHY':
            self.send_alert_email(saved_data)
        
        # Save image locally
        self.save_image_locally(image, detection_result)
        
        logger.info("✅ Detection cycle completed successfully")
        logger.info("=" * 60)
        return True
    
    def run(self):
        """Main monitoring loop"""
        logger.info("=" * 60)
        logger.info("Coffee Disease Monitor Starting")
        logger.info("=" * 60)
        
        # Load model
        if not self.load_model():
            logger.error("Cannot start without model")
            return
        
        # Initialize camera
        self.initialize_camera()
        
        # Authenticate
        if not self.authenticate():
            logger.error("Cannot start without authentication")
            return
        
        # Get interval
        interval_minutes = self.config['detection']['capture_interval_minutes']
        interval_seconds = interval_minutes * 60
        
        logger.info(f"Starting monitoring loop (interval: {interval_minutes} minutes)")
        logger.info(f"Farm: {self.config['farm']['farm_name']}")
        logger.info(f"Farmer: {self.config['farmer']['email']}")
        
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                logger.info(f"\nCycle #{cycle_count}")
                
                # Run detection
                self.run_detection_cycle()
                
                # Wait for next cycle
                logger.info(f"⏰ Waiting {interval_minutes} minutes until next detection...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("\n👋 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        
        if self.camera is not None:
            try:
                self.camera.stop()
                logger.info("Camera stopped")
            except:
                pass
        
        logger.info("Coffee Disease Monitor stopped")


def main():
    """Main entry point"""
    print("=" * 60)
    print("Coffee Disease Detection Monitor for Raspberry Pi")
    print("Updated with Direct API Integration")
    print("=" * 60)
    
    monitor = CoffeeDiseaseMonitor()
    monitor.run()


if __name__ == "__main__":
    main()