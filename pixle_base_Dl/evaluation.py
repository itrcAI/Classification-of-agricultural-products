from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Evaluate:
    def __init__(self, model, X_test, y_test, additional_features, label_encoder, file_path, X_cord, Y_cord, keep_labels):
        """
        Initialize the evaluation class with the model, test data, and parameters.

        :param model: Trained model for predictions
        :param X_test: Test features
        :param y_test: True labels (one-hot encoded or encoded)
        :param additional_features: Optional additional test features
        :param label_encoder: LabelEncoder for mapping labels
        :param file_path: Path to save evaluation results
        :param X_cord: X coordinates for visualization (optional)
        :param Y_cord: Y coordinates for visualization (optional)
        :param keep_labels: Labels to retain during reclassification
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.label_encoder = label_encoder
        self.file_path = file_path
        self.X_cord = X_cord
        self.Y_cord = Y_cord
        self.keep_labels = keep_labels
        self.additional_features_test = additional_features

    def evaluate_model(self):
        """
        Evaluate the model using accuracy, F1-score, Cohen's Kappa, and generate reports.
        Save results and visualizations to files.
        """
        # Predict on the test data
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(self.y_test, axis=1)

        # Calculate accuracy
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        # Calculate F1-Score
        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
        print(f"F1-Score (weighted): {f1:.2f}")

        # Ensure unique labels match between test set and predictions
        unique_labels = np.unique(np.concatenate([y_test_classes, y_pred_classes]))

        # Generate classification report
        class_report = classification_report(
            y_test_classes, y_pred_classes, labels=unique_labels,
            target_names=self.label_encoder.inverse_transform(unique_labels), output_dict=True
        )
        class_report_df = pd.DataFrame(class_report).transpose()

        # Compute confusion matrix
        conf_matrix = confusion_matrix(y_test_classes, y_pred_classes, labels=unique_labels, normalize='true')
        conf_matrix_df = pd.DataFrame(
            conf_matrix,
            index=self.label_encoder.inverse_transform(unique_labels),
            columns=self.label_encoder.inverse_transform(unique_labels)
        )

        # Plot and save the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_df, annot=True, fmt=".2f", cmap='Blues')
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        conf_matrix_path = self.file_path.replace('.xlsx', '_confusion_matrix.png')
        plt.savefig(conf_matrix_path)
        plt.close()
        print(f"Confusion matrix plot saved to {conf_matrix_path}")

        # Calculate Cohen's Kappa
        kappa = cohen_kappa_score(y_test_classes, y_pred_classes)
        print(f"Cohen's Kappa: {kappa:.2f}")

        # Save metrics and reports to Excel
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1-Score', 'Cohen\'s Kappa'],
            'Score': [accuracy, f1, kappa]
        })

        with pd.ExcelWriter(self.file_path) as writer:
            metrics_df.to_excel(writer, sheet_name='Overall Metrics', index=False)
            class_report_df.to_excel(writer, sheet_name='Classification Report')
            conf_matrix_df.to_excel(writer, sheet_name='Confusion Matrix')

        print(f"Metrics and classification report saved to {self.file_path}")

    def _reclassify_labels(self, y_pred):
        """
        Reclassify labels by mapping any label not in the keep_labels list to 'others'.

        :param y_pred: Predicted labels (one-hot encoded or encoded)
        :return: Reclassified true and predicted labels, updated label encoder
        """
        # Convert one-hot encoded labels to 1D array if necessary
        if len(y_pred.shape) > 1:
            self.y_test = np.argmax(self.y_test, axis=1)
            y_pred = np.argmax(y_pred, axis=1)

        # Get original label names
        original_labels_y_test = self.label_encoder.inverse_transform(self.y_test)
        original_labels_y_pred = self.label_encoder.inverse_transform(y_pred)

        # Reclassify labels to 'others' if not in keep_labels
        reclassified_labels_y_test = ['others' if label not in self.keep_labels else label for label in original_labels_y_test]
        reclassified_labels_y_pred = ['others' if label not in self.keep_labels else label for label in original_labels_y_pred]

        # Add 'others' to label encoder classes if not present
        if 'others' not in self.label_encoder.classes_:
            new_classes = np.append(self.label_encoder.classes_, 'others')
            self.label_encoder.classes_ = new_classes

        # Re-encode the reclassified labels
        reclassified_y_test = self.label_encoder.transform(reclassified_labels_y_test)
        reclassified_y_pred = self.label_encoder.transform(reclassified_labels_y_pred)

        return reclassified_y_test, reclassified_y_pred, self.label_encoder

    def evaluate_model_reclassified(self):
        """
        Evaluate the model after reclassifying labels into 'keep_labels' and 'others'.
        Save results and visualizations to files.
        """
        # Predict and reclassify labels
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        if self.keep_labels:
            self.y_test, y_pred_classes, updated_label_encoder = self._reclassify_labels(y_pred)

        # Compute confusion matrix with reclassified labels
        unique_labels = np.unique(self.y_test)

        if self.X_cord is not None and self.Y_cord is not None:
            predictions_df = pd.DataFrame({
                'label': updated_label_encoder.inverse_transform(y_pred_classes),
                'X': self.X_cord,
                'Y': self.Y_cord
            })
            predictions_csv_path = self.file_path.replace('.xlsx', '_predictions.csv')
            predictions_df.to_csv(predictions_csv_path, index=False)

        # Generate confusion matrix DataFrame
        conf_matrix = confusion_matrix(self.y_test, y_pred_classes, labels=unique_labels, normalize='true')
        reduced_classes = updated_label_encoder.inverse_transform(unique_labels)
        conf_matrix_df = pd.DataFrame(conf_matrix, index=reduced_classes, columns=reduced_classes)

        # Plot and save the reclassified confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_df, annot=True, fmt=".2f", cmap='Blues')
        plt.title('Normalized Confusion Matrix (Reclassified)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        conf_matrix_path = self.file_path.replace('.xlsx', '_reclassified_confusion_matrix.png')
        plt.savefig(conf_matrix_path)
        plt.close()
        print(f"Reclassified confusion matrix plot saved to {conf_matrix_path}")

        # Generate classification report
        class_report = classification_report(
            self.y_test, y_pred_classes, labels=unique_labels, 
            target_names=reduced_classes, output_dict=True
        )
        class_report_df = pd.DataFrame(class_report).transpose()

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred_classes)
        f1 = f1_score(self.y_test, y_pred_classes, average='weighted')
        kappa = cohen_kappa_score(self.y_test, y_pred_classes)

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1-Score', 'Cohen\'s Kappa'],
            'Score': [accuracy, f1, kappa]
        })

        # Save metrics and reports to Excel
        with pd.ExcelWriter(self.file_path) as writer:
            metrics_df.to_excel(writer, sheet_name='Overall Metrics', index=False)
            class_report_df.to_excel(writer, sheet_name='Classification Report')
            conf_matrix_df.to_excel(writer, sheet_name='Confusion Matrix')

        print(f"Reclassified evaluation saved to: {self.file_path}")
