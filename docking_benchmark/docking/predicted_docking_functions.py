import keras
from sklearn.model_selection import train_test_split


def create_two_layer_mlp(input_dim, hidden_dim, activation, optimizer, loss):
    model = keras.models.Sequential([
        keras.layers.Dense(hidden_dim, batch_input_shape=(None, input_dim), activation=activation),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer, loss=loss)
    model_grads = keras.backend.gradients(model.output, model.input)[0]
    grad_output_wrt_input = keras.backend.function([model.input], [model.output, model_grads])

    return model, grad_output_wrt_input


class MLPPredictedDockingScore:
    def __init__(self, dataset, *, input_dim, to_latent_fn, epochs=30, test_fraction=None, hidden_dim=1000,
                 activation='relu', optimizer='adam', loss='mse'):
        self.hidden_dim = hidden_dim
        self.dataset = dataset
        self.model, self._grad_fn = create_two_layer_mlp(input_dim, hidden_dim, activation, optimizer, loss)
        self.to_latent_fn = to_latent_fn
        self._fit(epochs, test_fraction)

    def _fit(self, epochs, test_fraction):
        x, y = self.dataset
        x = self.to_latent_fn(x)

        if test_fraction is not None:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_fraction)
            val_data = (x_test, y_test)
        else:
            x_train, y_train = x, y
            val_data = None

        self.model.fit(
            x_train, y_train,
            validation_data=val_data,
            epochs=epochs,
            verbose=True)

    def gradient(self, x):
        _, grad = self._grad_fn([x])
        return grad

    def latent_score(self, latent):
        return self.model.predict(latent).flatten().item()

    def smiles_score(self, smiles):
        return self.latent_score(self.to_latent_fn(smiles))
