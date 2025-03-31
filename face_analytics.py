import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix
import sqlite3

class FaceAnalytics:
    def __init__(self, db_path='auth_system.db', images_dir='registered_faces'):
        self.db_path = db_path
        self.images_dir = images_dir
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def get_registered_users(self):
        """Get list of registered users from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT user_id FROM users')
        users = cursor.fetchall()
        conn.close()
        return [user[0] for user in users]

    def analyze_face_features(self):
        """Analyze facial features of registered users"""
        users = self.get_registered_users()
        features_data = []
        
        for user_id in users:
            image_path = os.path.join(self.images_dir, f"{user_id}.jpg")
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    # Calculate various metrics
                    avg_intensity = np.mean(face_roi)
                    std_intensity = np.std(face_roi)
                    face_size = w * h
                    
                    features_data.append({
                        'user_id': user_id,
                        'face_size': face_size,
                        'avg_intensity': avg_intensity,
                        'std_intensity': std_intensity
                    })
        
        return pd.DataFrame(features_data)

    def plot_face_size_distribution(self):
        """Plot distribution of face sizes"""
        df = self.analyze_face_features()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='face_size', bins=20)
        plt.title('Distribution of Detected Face Sizes')
        plt.xlabel('Face Size (pixels²)')
        plt.ylabel('Count')
        plt.savefig('face_size_distribution.png')
        plt.close()

    def plot_intensity_analysis(self):
        """Plot face intensity analysis"""
        df = self.analyze_face_features()
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(y=df['avg_intensity'])
        plt.title('Average Face Intensity Distribution')
        plt.ylabel('Intensity Value')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df['std_intensity'])
        plt.title('Face Intensity Variation Distribution')
        plt.ylabel('Standard Deviation')
        
        plt.tight_layout()
        plt.savefig('face_intensity_analysis.png')
        plt.close()

    def analyze_verification_performance(self, num_tests=100):
        """Analyze verification performance with test cases"""
        users = self.get_registered_users()
        results = []
        
        for user_id in users:
            image_path = os.path.join(self.images_dir, f"{user_id}.jpg")
            if os.path.exists(image_path):
                stored_face = cv2.imread(image_path)
                
                # Test with various modifications
                for i in range(num_tests):
                    # Create test image with random modifications
                    test_face = stored_face.copy()
                    
                    # Add random brightness variation
                    brightness = np.random.uniform(0.8, 1.2)
                    test_face = cv2.convertScaleAbs(test_face, alpha=brightness)
                    
                    # Calculate similarity score
                    score = self.compare_faces(stored_face, test_face)
                    
                    results.append({
                        'user_id': user_id,
                        'test_num': i,
                        'score': score,
                        'brightness': brightness
                    })
        
        return pd.DataFrame(results)

    def compare_faces(self, face1, face2):
        """Compare two faces using template matching"""
        face1 = cv2.resize(face1, (256, 256))
        face2 = cv2.resize(face2, (256, 256))
        
        gray1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
        
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        return result[0][0]

    def plot_verification_performance(self):
        """Plot verification performance analysis"""
        df = self.analyze_verification_performance()
        
        plt.figure(figsize=(15, 5))
        
        # Score distribution
        plt.subplot(1, 3, 1)
        sns.histplot(data=df, x='score', bins=30)
        plt.axvline(x=0.8, color='r', linestyle='--', label='Threshold (0.8)')
        plt.title('Distribution of Similarity Scores')
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.legend()
        
        # Score vs Brightness
        plt.subplot(1, 3, 2)
        sns.scatterplot(data=df, x='brightness', y='score')
        plt.axhline(y=0.8, color='r', linestyle='--', label='Threshold (0.8)')
        plt.title('Similarity Score vs Brightness Variation')
        plt.xlabel('Brightness Factor')
        plt.ylabel('Similarity Score')
        plt.legend()
        
        # Performance by User
        plt.subplot(1, 3, 3)
        sns.boxplot(data=df, x='user_id', y='score')
        plt.axhline(y=0.8, color='r', linestyle='--', label='Threshold (0.8)')
        plt.title('Score Distribution by User')
        plt.xticks(rotation=45)
        plt.xlabel('User ID')
        plt.ylabel('Similarity Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('verification_performance.png')
        plt.close()

    def analyze_encryption_visualization(self):
        """Analyze and visualize original vs encrypted images"""
        users = self.get_registered_users()
        
        for user_id in users:
            image_path = os.path.join(self.images_dir, f"{user_id}.jpg")
            if os.path.exists(image_path):
                # Read original image
                original = cv2.imread(image_path)
                
                # Create encrypted version
                encrypted = self.create_encrypted_visualization(original)
                
                # Create comparison visualization
                plt.figure(figsize=(15, 5))
                
                # Original image
                plt.subplot(1, 3, 1)
                plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                plt.title('Original Image')
                plt.axis('off')
                
                # Encrypted image
                plt.subplot(1, 3, 2)
                plt.imshow(encrypted, cmap='gray')
                plt.title('Encrypted Visualization')
                plt.axis('off')
                
                # Difference visualization
                plt.subplot(1, 3, 3)
                diff = cv2.absdiff(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), encrypted)
                plt.imshow(diff, cmap='hot')
                plt.title('Difference Map')
                plt.axis('off')
                
                plt.suptitle(f'Image Encryption Analysis - User: {user_id}')
                plt.tight_layout()
                plt.savefig(f'encryption_visualization_{user_id}.png')
                plt.close()
                
                # Generate histogram comparison
                self.plot_encryption_histograms(original, encrypted, user_id)

    def create_encrypted_visualization(self, image):
        """Create a visualization of the encryption process"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold to simulate encryption
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        # Apply additional transformations to visualize encryption
        encrypted = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        encrypted = cv2.normalize(encrypted, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        return encrypted

    def plot_encryption_histograms(self, original, encrypted, user_id):
        """Plot histograms comparing original and encrypted images"""
        plt.figure(figsize=(15, 5))
        
        # Original image histogram
        plt.subplot(1, 3, 1)
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        plt.hist(original_gray.ravel(), 256, [0, 256], color='blue', alpha=0.7)
        plt.title('Original Image Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        
        # Encrypted image histogram
        plt.subplot(1, 3, 2)
        plt.hist(encrypted.ravel(), 256, [0, 256], color='red', alpha=0.7)
        plt.title('Encrypted Image Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        
        # Overlay comparison
        plt.subplot(1, 3, 3)
        plt.hist(original_gray.ravel(), 256, [0, 256], color='blue', alpha=0.5, label='Original')
        plt.hist(encrypted.ravel(), 256, [0, 256], color='red', alpha=0.5, label='Encrypted')
        plt.title('Histogram Comparison')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.suptitle(f'Histogram Analysis - User: {user_id}')
        plt.tight_layout()
        plt.savefig(f'encryption_histograms_{user_id}.png')
        plt.close()

    def generate_report(self):
        """Generate a comprehensive analysis report"""
        # Existing plots
        self.plot_face_size_distribution()
        self.plot_intensity_analysis()
        self.plot_verification_performance()
        
        # Add encryption visualization
        self.analyze_encryption_visualization()
        
        # Get feature statistics
        df_features = self.analyze_face_features()
        stats = df_features.describe()
        
        # Save statistics to CSV
        stats.to_csv('face_features_statistics.csv')
        
        # Create HTML report with added encryption analysis
        html_report = f"""
        <html>
        <head>
            <title>Face Authentication System Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                img {{ max-width: 100%; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Face Authentication System Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>1. Face Size Distribution</h2>
            <img src="face_size_distribution.png" alt="Face Size Distribution">
            
            <h2>2. Face Intensity Analysis</h2>
            <img src="face_intensity_analysis.png" alt="Face Intensity Analysis">
            
            <h2>3. Verification Performance</h2>
            <img src="verification_performance.png" alt="Verification Performance">
            
            <h2>4. Encryption Analysis</h2>
            {self._generate_encryption_section()}
            
            <h2>5. Statistical Summary</h2>
            {stats.to_html()}
        </body>
        </html>
        """
        
        with open('face_analysis_report.html', 'w') as f:
            f.write(html_report)

    def _generate_encryption_section(self):
        """Generate HTML for encryption analysis section"""
        users = self.get_registered_users()
        html = ""
        for user_id in users:
            html += f"""
            <h3>User: {user_id}</h3>
            <div>
                <img src="encryption_visualization_{user_id}.png" alt="Encryption Visualization">
                <img src="encryption_histograms_{user_id}.png" alt="Encryption Histograms">
            </div>
            """
        return html

if __name__ == "__main__":
    analyzer = FaceAnalytics()
    analyzer.generate_report() 