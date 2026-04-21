import os
import cv2
import numpy as np
from tqdm import tqdm
import weaviate
from sklearn.model_selection import train_test_split
from weaviate.classes.config import Configure, Property, DataType, VectorDistances

from faceProcessing import FaceProcessing
import kagglehub


# =========================
# 1. LOAD DATASET (RAW ONLY)
# =========================
path = kagglehub.dataset_download("hearfool/vggface2")
root = path
train_dir = os.path.join(root, "train")

X = []
y = []

label_map = {}
label_id = 0

print("Loading dataset...")

for person_id in tqdm(os.listdir(train_dir)):
    person_path = os.path.join(train_dir, person_id)

    if not os.path.isdir(person_path):
        continue

    if person_id not in label_map:
        label_map[person_id] = label_id
        label_id += 1

    label = label_map[person_id]

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # ❗ NO preprocessing — keep raw BGR image
        X.append(img)
        y.append(label)

X = np.array(X, dtype=object)  # important for variable-size images
y = np.array(y)

MAX_SAMPLES = 20000

indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)

X = np.array(X, dtype=object)[indices]
y = np.array(y)[indices]

names = {v: k for k, v in label_map.items()}

print("Classes:", len(names))
print("Dataset size:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 2. WEAVIATE SETUP
# =========================
CLASS_NAME = "Faces"

client = weaviate.connect_to_local()

if not client.collections.exists(CLASS_NAME):
    client.collections.create(
        name=CLASS_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE
        ),
        properties=[
            Property(name="label", data_type=DataType.TEXT)
        ]
    )

collection = client.collections.get(CLASS_NAME)


# =========================
# 3. FACE PROCESSOR
# =========================
processor = FaceProcessing()


# =========================
# 4. INDEX TRAINING DATA
# =========================
print("Indexing training data...")

for img, label in tqdm(list(zip(X_train, y_train))):

    # ❗ ALL preprocessing happens inside FaceProcessing
    results = processor.run(img)

    if not results:
        continue

    for face in results:
        collection.data.insert(
            properties={"label": str(label)},
            vector=face.embedding.tolist()
        )


# =========================
# 5. TESTING
# =========================
print("Evaluating...")

correct = 0
total = 0

for img, label in zip(X_test, y_test):

    results = processor.run(img)

    total += 1

    if not results:
        continue

    query_vec = np.mean(
        [f.embedding for f in results],
        axis=0
    )

    res = collection.query.near_vector(
        near_vector=query_vec.tolist(),
        limit=1
    )

    if res.objects:
        predicted = int(res.objects[0].properties["label"])

        if predicted == label:
            correct += 1

client.close()

# =========================
# 6. RESULTS
# =========================
print("Faces tested:", total)
print("Accuracy:", correct / total if total > 0 else 0)
print("Correct:", correct, "Total:", total)