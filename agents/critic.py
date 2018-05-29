from keras import layers, models, optimizers
from keras import backend as K

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        states0 = layers.Dense(units=400, kernel_regularizer=layers.regularizers.l2(0.0001))(states)
        states0 = layers.BatchNormalization()(states0)
        states0 = layers.Activation("relu")(states0)
        #states0 = layers.Dropout(0.3)(states0)
        states1 = layers.Dense(units=300, kernel_regularizer=layers.regularizers.l2(0.0001))(states0)
        states1 = layers.BatchNormalization()(states1)
        states1 = layers.Activation("relu")(states1) 
        #states1 = layers.Dropout(0.3)(states1)


        # Add hidden layer(s) for action pathway
        action0 = layers.Dense(units=400, kernel_regularizer=layers.regularizers.l2(0.01))(actions)
        action0 = layers.BatchNormalization()(action0)
        action0 = layers.Activation("relu")(action0)
        action1 = layers.Dense(units=300, kernel_regularizer=layers.regularizers.l2(0.01))(action0)
        action1 = layers.BatchNormalization()(action1)
        action1 = layers.Activation("relu")(action1) 

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.
        
        '''
        
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.4))

        '''

        # Combine state and action pathways
        net = layers.Add()([states1, action1])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)