import random as rnd
rnd.seed(42)

class Neuron:
    def __init__(self, n_entries: int, learning_rate: float = 0.1):
        self.n_entries = n_entries
        self.learning_rate = learning_rate

        self.w0 = rnd.uniform(-1, 1)  # bias
        self.x0 = 1

        self.weights = [rnd.uniform(-1, 1) for _ in range(n_entries)]

    def agregation(self, data: list):
        result = self.x0 * self.w0

        for x, w in zip(data, self.weights):
            result += x * w

        return result

    def activation(self, value):
        return 1 if value >= 0 else 0

    def predict(self, data: list):
        z = self.agregation(data)
        return (self.activation(z), z)

    def learn(self, data: list, expected_output: int, output: int, ponderate_sum: int):
        #error = expected_output - output
        error = expected_output - ponderate_sum

        self.w0 = self.w0 + self.learning_rate * error * self.x0

        for i in range(self.n_entries):
            self.weights[i] = self.weights[i] + self.learning_rate * error * data[i]


def train_perceptron(dataset, expected_output) -> Neuron:

    neuron = Neuron(len(dataset[0]), 0.01)

    for _ in range (100):
        errors = 0
        for i in range(0,len(dataset)):
            out, z = neuron.predict(dataset[i])
            print(neuron.weights)
            if expected_output[i] != out:
                errors += 1
            neuron.learn(dataset[i], expected_output[i], out, z)

        # if errors == 0: 
        #     break
    return neuron
    
def main():
    # dataset = [
    #     [0.2,0,1],
    #     [0.3,0,1],
    #     [0.4,1,1],
    #     [0.5,1,0],
    #     [0.6,1,0],
    #     [0.7,1,0],
    #     [0.8,1,0],
    #     [0.2,1,1],
    #     [0.9,1,0],
    #     [0.4,0,1],

    #     # exemplos extras importantes
    #     [0.7,0,0],
    #     [0.3,1,0],
    #     [0.3,0,0],
    #     [0.8,0,1],
    # ]

    # labels = [
    #     0,
    #     0,
    #     0,
    #     1,
    #     1,
    #     1,
    #     1,
    #     0,
    #     1,
    #     0,

    #     0,  # renda boa, mas sem emprego
    #     0,  # renda baixa, mesmo com emprego
    #     0,  # renda baixa
    #     0,  # dívida alta
    # ]
    # neuron: Neuron = train_perceptron(dataset, labels)
    
    # tests = [
    #     [1.0,1,0],
    #     [0.2,0,1],
    #     [0.5,1,1],
    #     [0.7,0,0],
    #     [0.3,1,0]
    # ]
    # for t in tests:
    #     print(t,neuron.predict(t))

    # [x,y]
    dataset = [

        [1,1],
        [2,2],
        [3,2],
        [4,3],
        [5,4],

        [1,5],
        [2,6],
        [3,7],
        [4,8],
        [5,9]
    ]

    # regra escondida:
    # y > x + 2

    dataset = [
        [0.1,0.1], [0.2,0.2], [0.3,0.2], [0.4,0.3], [0.5,0.4], [0.7,0.8], [0.8,0.9],
        [0.1,0.5], [0.2,0.6], [0.3,0.7], [0.4,0.8], [0.5,0.9], [0.7,1.0], [0.8,1.2]
    ]

    labels = [
        0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1
    ]

    neuron = train_perceptron(
        dataset,
        labels
    )

    print("\nPesos aprendidos:")
    print("bias:", neuron.w0)
    print("weights:", neuron.weights)

    tests = [
        [0.3,0.3],
        [0.3,0.6],
        [0.8,0.2],
        [0.7,1.0],
        [0.5,0.7],
        [1.0,0.5],
        [0.2,0.8],
    ]

    print("\nResultados:\n")

    for t in tests:

        pred, _ = neuron.predict(t)

        print(
            t,
            "->",
            pred
        )

if __name__ == '__main__':
    main()