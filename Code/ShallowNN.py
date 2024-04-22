# Imports not shown

# Load the dataset into a DataFrame
df = pd.read_csv("phishing.csv")

# Perform train-test split
kx = df.drop(columns=['Result'])  # Features
ky = df['Result']  # Target variable

kx_train, kx_test, ky_train, ky_test = train_test_split(kx, ky, test_size=0.25, random_state=42)

# Standardize features
scaler = StandardScaler()
kx_train_scaled = scaler.fit_transform(kx_train)
kx_test_scaled = scaler.transform(kx_test)

# Build the neural network model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(kx_train_scaled.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(kx_train_scaled, ky_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(kx_test_scaled, ky_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
