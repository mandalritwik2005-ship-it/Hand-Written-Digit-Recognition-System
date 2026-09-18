# Step 1: Import
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

# Step 2: Load MNIST
mnist = fetch_openml('mnist_784', version=1)

X = mnist.data
y = mnist.target.astype(int)

# Normalize (VERY IMPORTANT)
X = X / 255.0

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42
)

# Step 4: Train model (MLP - still ML)
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=20,
    verbose=True,
    random_state=42
)

model.fit(X_train, y_train)

# Step 5: Accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)