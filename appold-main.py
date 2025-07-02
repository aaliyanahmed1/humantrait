# This model accepts input images of size 224x224 pixels, which is handled by the transform pipeline
import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import logging
import random
from torchvision import models, transforms
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define model paths
FAIRFACE_MODEL_PATH = "fairface.pt"


class UnifiedTraitPredictor:
    def __init__(self):
        # Create directories
        os.makedirs("models", exist_ok=True)

        # Paths
        self.fairface_model_path = FAIRFACE_MODEL_PATH
        self.output_folder = "output"
        self.output_images_folder = os.path.join(self.output_folder, "processed_images")
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.output_images_folder, exist_ok=True)

        # Labels
        self.race_labels = ["White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian",
                            "Middle Eastern"]
        self.gender_labels = ["Male", "Female"]
        self.age_labels = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load models
        self.load_fairface_model()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_fairface_model(self):
        """Load the FairFace model for race, gender, and age prediction"""
        try:
            # Load FairFace model
            self.model_fairface = models.resnet34(pretrained=False)
            self.model_fairface.fc = nn.Linear(self.model_fairface.fc.in_features,
                                               18)  # 7 (Race) + 2 (Gender) + 9 (Age)
            self.model_fairface.load_state_dict(torch.load(self.fairface_model_path, map_location=self.device))
            self.model_fairface = self.model_fairface.to(self.device)
            self.model_fairface.eval()
            logger.info("✅ FairFace model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FairFace model: {str(e)}")
            raise

    def detect_faces(self, image):
        """Detect faces in an image using OpenCV's Haar Cascade"""
        faces = []
        h, w = image.shape[:2]

        # Load OpenCV's face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 and image.shape[2] == 3 else image

        # Detect faces
        detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(detections) > 0:
            for (x, y, width, height) in detections:
                # Ensure coordinates are valid
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)

                if width > 0 and height > 0:
                    faces.append((x, y, width, height))

        return faces

    def preprocess_for_fairface(self, face):
        """Preprocess face image for FairFace model"""
        try:
            return self.transform(face).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing image for FairFace: {str(e)}")
            return None

    def predict_fairface(self, face_tensor):
        """Predict race, gender, and age using FairFace model"""
        try:
            with torch.no_grad():
                output = self.model_fairface(face_tensor).cpu().numpy().squeeze()

            race_pred = self.race_labels[np.argmax(output[:7])]
            race_conf = float(np.max(output[:7]))

            gender_pred = self.gender_labels[np.argmax(output[7:9])]
            gender_conf = float(np.max(output[7:9]))

            age_pred = self.age_labels[np.argmax(output[9:18])]

            return race_pred, race_conf, gender_pred, gender_conf, age_pred
        except Exception as e:
            logger.error(f"Error during FairFace prediction: {str(e)}")
            return None, None, None, None, None

    def age_to_numeric(self, age_range):
        """Convert age range to numeric value (using midpoint)"""
        if age_range == "0-2":
            return 1
        elif age_range == "3-9":
            return 6
        elif age_range == "10-19":
            return 15
        elif age_range == "20-29":
            return 25
        elif age_range == "30-39":
            return 35
        elif age_range == "40-49":
            return 45
        elif age_range == "50-59":
            return 55
        elif age_range == "60-69":
            return 65
        elif age_range == "70+":
            return 75
        return 25  # Default if parsing fails

    def calculate_intelligence(self, age):
        """Calculate intelligence based on age"""
        # Base intelligence value
        intelligence = 0.5

        # Add age influence
        intelligence += (age / 100)

        # Cap between 0-1
        return min(max(intelligence, 0.0), 1.0)

    def calculate_confidence(self, gender, age):
        """Calculate confidence based on gender and age"""
        # Base confidence value
        confidence = 0.5

        # Gender influence
        if gender == "Male":
            confidence += 0.05
        elif gender == "Female":
            confidence -= 0.05

        # Age influence (younger = more confident)
        confidence -= (age / 150)

        # Cap between 0-1
        return min(max(confidence, 0.0), 1.0)

    def calculate_cooperativeness(self, race, age):
        """Calculate cooperativeness based on race and age"""
        # Base cooperativeness value
        cooperativeness = 0.5

        # Race influence
        race_baseline = 1.0  # Default
        if race == "White":
            race_baseline = 1.0
        elif race == "Black":
            race_baseline = 0.9
        elif race in ["East Asian", "Southeast Asian"]:
            race_baseline = 0.95
        else:  # Other races
            race_baseline = 0.92

        cooperativeness *= race_baseline

        # Age influence
        cooperativeness += (age / 120)

        # Cap between 0-1
        return min(max(cooperativeness, 0.0), 1.0)

    def calculate_celibacy(self, age, gender):
        """Calculate celibacy probability based on age and gender"""
        # Base celibacy value
        celibacy = 0.3

        # Age influence (older = more likely to be celibate)
        if age < 20:
            celibacy += 0.3  # Young people more likely to be celibate
        elif age > 60:
            celibacy += 0.2  # Older people somewhat more likely to be celibate
        else:
            celibacy -= 0.1  # Middle-aged less likely to be celibate

        # Gender influence
        if gender == "Male":
            celibacy -= 0.05
        elif gender == "Female":
            celibacy += 0.05

        # Add some randomness
        celibacy += random.uniform(-0.1, 0.1)

        # Cap between 0-1
        return min(max(celibacy, 0.0), 1.0)

    def calculate_attractiveness(self, age, gender, face_shape=None):
        """Calculate face attractiveness based on age, gender and face features"""
        # Base attractiveness
        attractiveness = 0.5

        # Age influence (prime age = more attractive)
        if 20 <= age <= 35:
            attractiveness += 0.2
        elif age > 60:
            attractiveness -= 0.15
        elif age < 18:
            attractiveness -= 0.1

        # Add some randomness (genetic lottery)
        attractiveness += random.uniform(-0.2, 0.2)

        # Cap between 0-1
        return min(max(attractiveness, 0.0), 1.0)

    def calculate_big_spender(self, age, race):
        """Calculate big spender trait based on age and race"""
        # Base value
        big_spender = 0.4

        # Age influence (middle-aged tend to spend more)
        if 30 <= age <= 55:
            big_spender += 0.2
        elif age < 20:
            big_spender -= 0.2

        # Race influence (based on economic stereotypes)
        if race in ["White", "East Asian"]:
            big_spender += 0.1

        # Add randomness
        big_spender += random.uniform(-0.15, 0.15)

        # Cap between 0-1
        return min(max(big_spender, 0.0), 1.0)

    def calculate_presentable(self, age, gender):
        """Calculate how presentable someone appears"""
        # Base value
        presentable = 0.5

        # Age influence (middle-aged tend to be more presentable)
        if 30 <= age <= 60:
            presentable += 0.15
        elif age < 20:
            presentable -= 0.1

        # Gender influence
        if gender == "Female":
            presentable += 0.1

        # Add randomness
        presentable += random.uniform(-0.1, 0.1)

        # Cap between 0-1
        return min(max(presentable, 0.0), 1.0)

    def calculate_muscle_percentage(self, age, gender):
        """Calculate approximate muscle percentage based on age and gender"""
        # Base muscle percentage
        if gender == "Male":
            base_muscle = 0.4  # 40% for males
        else:
            base_muscle = 0.3  # 30% for females

        # Age influence (peaks at 25-35, declines after)
        if 25 <= age <= 35:
            age_factor = 0.05
        elif age < 25:
            age_factor = 0.03
        else:
            age_factor = -0.05 * ((age - 35) / 20)  # Gradual decline

        muscle = base_muscle + age_factor

        # Add randomness
        muscle += random.uniform(-0.08, 0.08)

        # Cap between 0.15-0.6
        return min(max(muscle, 0.15), 0.6)

    def calculate_fat_percentage(self, age, gender):
        """Calculate approximate fat percentage based on age and gender"""
        # Base fat percentage
        if gender == "Male":
            base_fat = 0.18  # 18% for males
        else:
            base_fat = 0.25  # 25% for females

        # Age influence (increases with age)
        age_factor = 0.002 * max(0, age - 25)

        fat = base_fat + age_factor

        # Add randomness
        fat += random.uniform(-0.05, 0.1)

        # Cap between 0.1-0.45
        return min(max(fat, 0.1), 0.45)

    def calculate_dominance(self, gender, age, muscle_percentage):
        """Calculate dominance trait based on gender, age and muscle percentage"""
        # Base dominance
        dominance = 0.5

        # Gender influence
        if gender == "Male":
            dominance += 0.1
        else:
            dominance -= 0.05

        # Age influence (peaks at middle age)
        if 30 <= age <= 50:
            dominance += 0.1
        elif age < 25 or age > 65:
            dominance -= 0.1

        # Muscle influence
        dominance += (muscle_percentage - 0.3) * 0.5

        # Add randomness
        dominance += random.uniform(-0.1, 0.1)

        # Cap between 0-1
        return min(max(dominance, 0.0), 1.0)

    def calculate_power(self, age, gender, dominance):
        """Calculate power trait based on age, gender and dominance"""
        # Base power
        power = 0.4

        # Age influence (peaks at middle age)
        if 40 <= age <= 60:
            power += 0.2
        elif age < 30:
            power -= 0.1

        # Gender influence
        if gender == "Male":
            power += 0.05

        # Dominance influence
        power += dominance * 0.3

        # Add randomness
        power += random.uniform(-0.1, 0.1)

        # Cap between 0-1
        return min(max(power, 0.0), 1.0)

    def create_results_panel(self, results, panel_width=400, line_height=30):
        """Create a separate panel with results instead of overlaying on image"""
        # Calculate panel height based on number of text lines
        text_lines = [
            f"Race: {results['race']} ({results['race_confidence']:.2f})",
            f"Gender: {results['gender']} ({results['gender_confidence']:.2f})",
            f"Age: {results['age']}",
            f"Intelligence: {results['intelligence']:.3f}",
            f"Confidence: {results['confidence']:.3f}",
            f"Cooperativeness: {results['cooperativeness']:.3f}",
            f"Celibacy: {results['celibacy']:.3f}",
            f"Attractiveness: {results['attractiveness']:.3f}",
            f"Big Spender: {results['big_spender']:.3f}",
            f"Presentable: {results['presentable']:.3f}",
            f"Muscle %: {results['muscle_percentage']:.3f}",
            f"Fat %: {results['fat_percentage']:.3f}",
            f"Dominance: {results['dominance']:.3f}",
            f"Power: {results['power']:.3f}"
        ]

        panel_height = len(text_lines) * line_height + 40  # Add some padding

        # Create a white panel
        panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255

        # Add title
        cv2.putText(
            panel,
            "TRAIT ANALYSIS",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        # Add horizontal line
        cv2.line(panel, (20, 40), (panel_width - 20, 40), (0, 0, 0), 1)

        # Add text lines
        for i, line in enumerate(text_lines):
            y_pos = 70 + i * line_height
            cv2.putText(
                panel,
                line,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

            # For traits with ranges, add a visual bar
            if i >= 3:  # Skip race, gender, age
                trait_name, trait_value_str = line.split(": ")
                trait_value = float(trait_value_str)

                # Draw bar background
                bar_x = 250
                bar_width = 120
                bar_height = 15
                bar_y = y_pos - 12

                cv2.rectangle(panel,
                              (bar_x, bar_y),
                              (bar_x + bar_width, bar_y + bar_height),
                              (200, 200, 200),
                              -1)

                # Draw filled portion of bar
                filled_width = int(bar_width * trait_value)
                cv2.rectangle(panel,
                              (bar_x, bar_y),
                              (bar_x + filled_width, bar_y + bar_height),
                              (0, 120, 255),
                              -1)

                # Draw bar outline
                cv2.rectangle(panel,
                              (bar_x, bar_y),
                              (bar_x + bar_width, bar_y + bar_height),
                              (0, 0, 0),
                              1)

        return panel

    def process_images(self, input_folder):
        """Process all images in the input folder"""
        results = []
        image_paths = [os.path.join(input_folder, img) for img in os.listdir(input_folder)
                       if img.lower().endswith((".jpg", ".jpeg", ".png"))]

        if not image_paths:
            logger.warning(f"No images found in {input_folder}")
            return results

        logger.info(f"Found {len(image_paths)} images to process")

        for img_path in tqdm(image_paths, desc="Processing Images"):
            try:
                filename = os.path.basename(img_path)
                image = cv2.imread(img_path)

                if image is None:
                    logger.warning(f"Could not read image: {img_path}")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Detect faces using OpenCV
                detections = self.detect_faces(image_rgb)

                if not detections:
                    logger.warning(f"No faces detected in {img_path}")
                    # Skip images with no faces detected
                    continue

                # Process the first detected face (primary face)
                x, y, w, h = detections[0]

                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, image_rgb.shape[1] - x)
                h = min(h, image_rgb.shape[0] - y)

                face = image_rgb[y:y + h, x:x + w]

                if face.size == 0:
                    logger.warning(f"Invalid face region in {img_path}")
                    continue

                # Preprocess for FairFace model
                face_tensor_fairface = self.preprocess_for_fairface(face)

                if face_tensor_fairface is None:
                    logger.warning(f"Failed to preprocess face in {img_path}")
                    continue

                # Make predictions
                race_pred, race_conf, gender_pred, gender_conf, age_pred = self.predict_fairface(face_tensor_fairface)

                if not all([race_pred, gender_pred, age_pred]):
                    logger.warning(f"Prediction failed for {img_path}")
                    continue

                # Convert age to numeric value for calculations
                age_numeric = self.age_to_numeric(age_pred)

                # Calculate derived traits
                intelligence = self.calculate_intelligence(age_numeric)
                confidence = self.calculate_confidence(gender_pred, age_numeric)
                cooperativeness = self.calculate_cooperativeness(race_pred, age_numeric)

                # Calculate new traits
                celibacy = self.calculate_celibacy(age_numeric, gender_pred)
                attractiveness = self.calculate_attractiveness(age_numeric, gender_pred)
                big_spender = self.calculate_big_spender(age_numeric, race_pred)
                presentable = self.calculate_presentable(age_numeric, gender_pred)
                muscle_percentage = self.calculate_muscle_percentage(age_numeric, gender_pred)
                fat_percentage = self.calculate_fat_percentage(age_numeric, gender_pred)
                dominance = self.calculate_dominance(gender_pred, age_numeric, muscle_percentage)
                power = self.calculate_power(age_numeric, gender_pred, dominance)

                # Create result dictionary
                result_data = {
                    "filename": filename,
                    "age": age_pred,
                    "gender": gender_pred,
                    "gender_confidence": round(gender_conf, 3),
                    "race": race_pred,
                    "race_confidence": round(race_conf, 3),
                    "intelligence": round(intelligence, 3),
                    "confidence": round(confidence, 3),
                    "cooperativeness": round(cooperativeness, 3),
                    "celibacy": round(celibacy, 3),
                    "attractiveness": round(attractiveness, 3),
                    "big_spender": round(big_spender, 3),
                    "presentable": round(presentable, 3),
                    "muscle_percentage": round(muscle_percentage, 3),
                    "fat_percentage": round(fat_percentage, 3),
                    "dominance": round(dominance, 3),
                    "power": round(power, 3)
                }

                # Add result to our list
                results.append(result_data)

                # Create a copy of the image and draw rectangle around the face
                output_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Create results panel
                results_panel = self.create_results_panel(result_data)

                # Combine image and panel side by side
                img_height, img_width = output_img.shape[:2]
                panel_height, panel_width = results_panel.shape[:2]

                # Resize image if it's too tall compared to panel
                if img_height > panel_height * 1.5:
                    scale_factor = panel_height / img_height
                    new_width = int(img_width * scale_factor)
                    output_img = cv2.resize(output_img, (new_width, panel_height))
                    img_height, img_width = output_img.shape[:2]

                # Create combined image
                combined_width = img_width + panel_width
                combined_height = max(img_height, panel_height)
                combined_img = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255

                # Place image and panel
                combined_img[0:img_height, 0:img_width] = output_img
                combined_img[0:panel_height, img_width:img_width + panel_width] = results_panel

                # Save combined image
                output_path = os.path.join(self.output_images_folder, f"processed_{filename}")
                cv2.imwrite(output_path, combined_img)
                logger.debug(f"Saved processed image: {output_path}")

            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")

        # Save results
        self.save_results(results)
        return results

    def save_results(self, results):
        """Save results to CSV and JSON"""
        try:
            if not results:
                logger.warning("No results to save")
                return

            # Save to CSV
            csv_path = os.path.join(self.output_folder, "predictions.csv")
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)
            logger.info(f"✅ Results saved to CSV: {csv_path}")

            # Save to JSON
            json_path = os.path.join(self.output_folder, "predictions.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"✅ Results saved to JSON: {json_path}")

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")


def main():
    try:
        # Set input folder
        input_folder = "test-1"  # Update with your test folder path

        # Check if input folder exists
        if not os.path.exists(input_folder):
            logger.error(f"Input folder not found: {input_folder}")
            logger.info("Creating test-images folder. Please add images there.")
            os.makedirs(input_folder, exist_ok=True)
            return

        # Initialize predictor
        predictor = UnifiedTraitPredictor()

        # Process images
        logger.info(f"Starting image processing from folder: {input_folder}")
        results = predictor.process_images(input_folder)

        if results:
            logger.info(f"Successfully processed {len(results)} images")
            logger.info(f"Processed images saved in: {predictor.output_images_folder}")
        else:
            logger.warning("No results were generated")

    except Exception as e:
        logger.error(f"An error occurred in the main function: {str(e)}")


if __name__ == "__main__":
    main()