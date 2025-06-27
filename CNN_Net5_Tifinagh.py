import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os
from PIL import Image
from sklearn.model_selection import train_test_split

class LeNet5:
    def __init__(self, input_shape=(32, 32, 1), num_classes=33):
        self.input_shape = input_shape
        self.num_classes = num_classes
        assert len(input_shape) == 3 and input_shape[0] == 32 and input_shape[1] == 32, \
            "L'entrée doit être de forme (32, 32, canaux)"
        
        # Initialisation des paramètres
        self.params = {}
        
        # Couche C1: 6 filtres 5x5x1
        self.params['C1_filters'] = np.random.randn(6, 5, 5, 1) * 0.1
        
        # Couche C3: 16 filtres 5x5x6
        self.params['C3_filters'] = np.random.randn(16, 5, 5, 6) * 0.1
        
        # Couche C5: 120 filtres 5x5x16
        self.params['C5_filters'] = np.random.randn(120, 5, 5, 16) * 0.1
        
        # Couche F6: 84 neurones
        self.params['F6_weights'] = np.random.randn(120, 84) * 0.1
        self.params['F6_bias'] = np.zeros(84)
        
        # Couche de sortie
        self.params['output_weights'] = np.random.randn(84, num_classes) * 0.1
        self.params['output_bias'] = np.zeros(num_classes)
        
        self.cache = {}
        self.m = {}  # Pour Adam optimizer
        self.v = {}  # Pour Adam optimizer
        self.t = 0   # Pour Adam optimizer

    def tanh(self, x):
        assert isinstance(x, np.ndarray), "Entrée tanh doit être un array"
        return np.tanh(x)
    
    def tanh_deriv(self, x):
        assert isinstance(x, np.ndarray), "Entrée tanh_deriv doit être un array"
        return 1 - np.tanh(x)**2
    
    def softmax(self, x):
        assert len(x.shape) == 2, "Entrée softmax doit être de forme (batch_size, num_classes)"
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def average_pooling(self, x, pool_size=2, stride=2):
        n, h, w, c = x.shape
        assert h >= pool_size and w >= pool_size, "Dimensions insuffisantes pour pooling"
        h_out = (h - pool_size) // stride + 1
        w_out = (w - pool_size) // stride + 1
        
        output = np.zeros((n, h_out, w_out, c))
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                h_end = h_start + pool_size
                w_start = j * stride
                w_end = w_start + pool_size
                
                pool_region = x[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = np.mean(pool_region, axis=(1, 2))
        
        return output
    
    def conv2d(self, x, filters, stride=1):
        n, h, w, c = x.shape
        f_num, fh, fw, fc = filters.shape
        assert c == fc, "Nombre de canaux incompatible"
        
        h_out = (h - fh) // stride + 1
        w_out = (w - fw) // stride + 1
        
        output = np.zeros((n, f_num, h_out, w_out))
        
        for i in range(n):
            for f in range(f_num):
                for hi in range(h_out):
                    for wi in range(w_out):
                        h_start = hi * stride
                        h_end = h_start + fh
                        w_start = wi * stride
                        w_end = w_start + fw
                        
                        region = x[i, h_start:h_end, w_start:w_end, :]
                        output[i, f, hi, wi] = np.sum(region * filters[f])
        
        return output
    
    def forward(self, x):
        assert x.ndim == 4 and x.shape[1:] == self.input_shape, "Entrée incorrecte pour le modèle"
        self.cache['input'] = x
        
        # Couche C1
        x = self.conv2d(x, self.params['C1_filters'])
        x = self.tanh(x)
        self.cache['C1_output'] = x
        
        # Reshape pour pooling
        x = x.transpose(0, 2, 3, 1)
        
        # Couche S2
        x = self.average_pooling(x)
        self.cache['S2_output'] = x
        
        # Couche C3
        x = self.conv2d(x, self.params['C3_filters'])
        x = self.tanh(x)
        self.cache['C3_output'] = x
        
        # Reshape pour pooling
        x = x.transpose(0, 2, 3, 1)
        
        # Couche S4
        x = self.average_pooling(x)
        self.cache['S4_output'] = x
        
        # Couche C5
        x = self.conv2d(x, self.params['C5_filters'])
        x = self.tanh(x)
        self.cache['C5_output'] = x
        
        # Flatten
        x = x.reshape(x.shape[0], -1)
        
        # Couche F6
        x = np.dot(x, self.params['F6_weights']) + self.params['F6_bias']
        x = self.tanh(x)
        self.cache['F6_output'] = x
        
        # Couche de sortie
        x = np.dot(x, self.params['output_weights']) + self.params['output_bias']
        x = self.softmax(x)
        self.cache['output'] = x
        
        return x
    
    def backward(self, x, y, output):
        assert x.shape[0] == y.shape[0] == output.shape[0], "Batch size incompatible"
        assert output.shape[1] == self.num_classes, "Nombre de classes incorrect"
        # Gradient de la loss
        dout = output - y
        
        # Couche de sortie
        self.grads = {}
        self.grads['output_weights'] = np.dot(self.cache['F6_output'].T, dout)
        self.grads['output_bias'] = np.sum(dout, axis=0)
        
        # Couche F6
        dF6 = np.dot(dout, self.params['output_weights'].T) * self.tanh_deriv(self.cache['F6_output'])
        self.grads['F6_weights'] = np.dot(self.cache['C5_output'].reshape(x.shape[0], -1).T, dF6)
        self.grads['F6_bias'] = np.sum(dF6, axis=0)
        
        # Couche C5
        dC5 = np.dot(dF6, self.params['F6_weights'].T).reshape(self.cache['C5_output'].shape)
        dC5 = dC5 * self.tanh_deriv(self.cache['C5_output'])
        
        # ... (le reste de la backprop à implémenter de manière similaire)
        
    def update_params(self, optimizer, learning_rate):
        assert optimizer in ['sgd', 'adam'], "Optimiseur non reconnu"
        assert learning_rate > 0, "Learning rate doit être positif"
        if optimizer == 'sgd':
            for param in self.params:
                self.params[param] -= learning_rate * self.grads.get(param, 0)
        elif optimizer == 'adam':
            self.t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            
            for param in self.params:
                if param not in self.m:
                    self.m[param] = np.zeros_like(self.params[param])
                    self.v[param] = np.zeros_like(self.params[param])
                
                self.m[param] = beta1 * self.m[param] + (1 - beta1) * self.grads.get(param, 0)
                self.v[param] = beta2 * self.v[param] + (1 - beta2) * (self.grads.get(param, 0)**2)
                
                m_hat = self.m[param] / (1 - beta1**self.t)
                v_hat = self.v[param] / (1 - beta2**self.t)
                
                self.params[param] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
    
    def compute_loss(self, y_true, y_pred):
        assert y_true.shape == y_pred.shape, "Dimensions mismatch pour la loss"
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(y_true * np.log(y_pred))
    
    def compute_accuracy(self, y_true, y_pred):
        assert y_true.shape == y_pred.shape, "Dimensions mismatch pour accuracy"
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, learning_rate=0.01, optimizer='sgd'):
        assert X_train.shape[0] == y_train.shape[0], "Taille des données d'entraînement incorrecte"
        assert X_val.shape[0] == y_val.shape[0], "Taille des données de validation incorrecte"
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []
        
        for epoch in range(epochs):
            # Shuffle
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            epoch_acc = 0
            
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward
                output = self.forward(X_batch)
                
                # Loss et accuracy
                batch_loss = self.compute_loss(y_batch, output)
                batch_acc = self.compute_accuracy(y_batch, output)
                
                epoch_loss += batch_loss * X_batch.shape[0]
                epoch_acc += batch_acc * X_batch.shape[0]
                
                # Backward
                self.backward(X_batch, y_batch, output)
                
                # Update
                self.update_params(optimizer, learning_rate)
            
            # Métriques d'epoch
            epoch_loss /= X_train.shape[0]
            epoch_acc /= X_train.shape[0]
            
            # Validation
            val_output = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_output)
            val_acc = self.compute_accuracy(y_val, val_output)
            
            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        return train_loss_history, train_acc_history, val_loss_history, val_acc_history

# Fonctions utilitaires restantes (load_tifinagh_dataset, prepare_data, plot_confusion_matrix, etc.)
def load_tifinagh_dataset(data_path='amhcd-data-64/tifinagh-images/'):
    assert os.path.exists(data_path), "Chemin des données invalide"
    images = []
    labels = []
    class_names = sorted(os.listdir(data_path))
    
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        for img_file in os.listdir(class_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path).convert('L')
                img = img.resize((32, 32))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(class_name)
    
    assert len(images) > 0, "Aucune image chargée"
    X = np.array(images).reshape(-1, 32, 32, 1)
    y = label_encoder.transform(labels)
    
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1, 1)).toarray()
    
    return X, y, label_encoder.classes_

def prepare_data():
    X, y, class_names = load_tifinagh_dataset()
    assert X.ndim == 4 and y.ndim == 2, "Données mal formatées"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names

def plot_history(train_loss, train_acc, val_loss, val_acc):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    assert len(y_true) == len(y_pred), "y_true et y_pred doivent avoir la même taille"
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Exécution principale
if __name__ == "__main__":
    # Chargement des données
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = prepare_data()
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Classes: {class_names}")
    
    assert len(class_names) == y_train.shape[1], "Nombre de classes incohérent"
    
    # Création du modèle
    model = LeNet5(input_shape=(32, 32, 1), num_classes=len(class_names))
    
    # Entraînement
    train_loss, train_acc, val_loss, val_acc = model.train(
        X_train, y_train, X_val, y_val,
        epochs=15,
        batch_size=32,
        learning_rate=0.001,
        optimizer='adam'
    )
    
    # Visualisation
    plot_history(train_loss, train_acc, val_loss, val_acc)
    
    # Évaluation
    test_output = model.forward(X_test)
    assert test_output.shape[0] == y_test.shape[0], "Résultat du test incompatible"
    test_acc = model.compute_accuracy(y_test, test_output)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Matrice de confusion
    y_pred = np.argmax(test_output, axis=1)
    y_true = np.argmax(y_test, axis=1)
    plot_confusion_matrix(y_true, y_pred, class_names)
