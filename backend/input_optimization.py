import tensorflow as tf

class input_optimization():
    def __init__(self, model, X_input, learning_rate = 0.01, num_iterations = 1000): 
        self.model = model       # the model
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.X_input = X_input
        
    def loss_function(self,y_pred, y_target):
        return tf.reduce_mean(tf.square(y_pred - y_target))
    
    def optimize(self, target):
        for i in range(self.num_iterations):
            with tf.GradientTape() as tape:
                tape.watch(self.X_input)
                y_pred = self.model(self.X_input)
                loss = self.loss_function(y_pred, target)

            gradients = tape.gradient(loss, self.X_input)
            self.X_input = self.X_input - self.learning_rate * gradients
        return self.X_input