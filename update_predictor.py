import sys

def replace_model():
    with open('src/predictor_ai.py', 'r') as f:
        content = f.read()

    # Replace imports
    imports_to_add = """from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, LayerNormalization, Add, Activation, Attention, GlobalAveragePooling1D
"""
    content = content.replace("from tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import Dense, GRU", imports_to_add)

    # Rename class
    content = content.replace("class GRUPredictor:", "class AttnBiLSTMPredictor:")
    content = content.replace("GRU-based recurrent", "Attention-Bi-LSTM based recurrent")
    content = content.replace("Uses a GRU", "Uses an Attention-Bi-LSTM mechanism")
    
    # Replace build_model
    old_build_model = """    def build_model(self, feature_dim):
        model = Sequential()
        # Recurrent backbone: GRU (replaces previous LSTM)
        model.add(GRU(32, return_sequences=False, input_shape=(self.window_size, feature_dim)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.model = model"""

    new_build_model = """    def build_model(self, feature_dim):
        inputs = Input(shape=(self.window_size, feature_dim))
        
        # 1. Attention Mechanism (Self-Attention)
        attn_out = Attention()([inputs, inputs])
        
        # 2. Layer Normalization
        norm_out = LayerNormalization()(attn_out)
        
        # 3. Bi-LSTM Layer
        bilstm_out = Bidirectional(LSTM(16, return_sequences=True))(norm_out)
        
        # 4. Residual Connection with Tanh
        projected_norm = Dense(32)(norm_out)
        res_out = Add()([bilstm_out, projected_norm])
        res_out = Activation('tanh')(res_out)
        
        # 5. Regression Output
        pooled_out = GlobalAveragePooling1D()(res_out)
        fc1 = Dense(16, activation='relu')(pooled_out)
        outputs = Dense(1)(fc1)
        
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.model = model"""

    content = content.replace(old_build_model, new_build_model)
    
    # Fix main block
    content = content.replace("predictor = GRUPredictor(", "predictor = AttnBiLSTMPredictor(")
    
    with open('src/predictor_ai.py', 'w') as f:
        f.write(content)

replace_model()
print("Updated predictor_ai.py")
