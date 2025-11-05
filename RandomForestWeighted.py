import numpy as np
from scipy import stats
import random
from collections import Counter
from scipy.stats import entropy


'''def weights_calculate(y):
    class_count = Counter(y) # Conta o número de amostras por classe
    samples= sum(class_count.values()) #calcula o número total de amostras
    #cria os pesos- para cada classe cls, o peso é samples/class_count[cls]
    weights = {
        cls: samples/class_count[cls]
        for cls in sorted(class_count)
    }

    #calcula a soma total dos pesos
    total_weight = sum(weights.values())
    n_classes= len(weights)
    #normaliza os pesos para que a media seja 1 e ajusta o peso consoante isso para cada classe cls
    for cls in weights:
        weights[cls]= weights[cls] * n_classes / total_weight
    return weights'''


def adaptive_weights_calculate(y, confidence_factor=0.70):
    """
    Calcula pesos adaptativos que consideram tanto a frequência das classes
    quanto um fator de confiança para equilibrar o desempenho

    confidence_factor controla quanto o modelo deve favorecer a classe minoritária
    """
    #igual à funcao de cima
    class_count = Counter(y)
    samples = sum(class_count.values())
    base_weights = {
        cls: samples/class_count[cls]
        for cls in sorted(class_count)
    }

    # Identifica a classe minoritária(com menos amostras)e a maioritaria(com mais)
    minority_class = min(class_count, key=class_count.get)
    majority_class = max(class_count, key=class_count.get)

    #cria copia dos pesos base para modificar
    adaptive_weights = base_weights.copy()
    #aplica o fator de confiança à classe minoritária
    #isso significa que a classe minoritária terá um peso maior, mas controlado pelo fator de confianca
    if minority_class in adaptive_weights:
        adaptive_weights[minority_class] = base_weights[minority_class] * confidence_factor

    #igual à funcao de cima
    total_weight = sum(adaptive_weights.values())
    n_classes = len(adaptive_weights)
    for cls in adaptive_weights:
        adaptive_weights[cls] = adaptive_weights[cls] * n_classes / total_weight
    return adaptive_weights


def weighted_entropy(p, weights):
    """
    calcula a entropia ponderada, dando mais importância às classes com maior peso.
    """
    class_counts = np.bincount(p)
    class_probs = class_counts / np.sum(class_counts)

    # Multiplicar as probabilidades de classe pelos pesos correspondentes
    weighted_probs = np.array([class_probs[i] * weights.get(i, 1) for i in range(len(class_probs))])
    weighted_probs /= np.sum(weighted_probs)  # Normalização

    return entropy(weighted_probs)


def balanced_information_gain(y, splits, weights):
    """
    calcula o ganho de informação de uma divisão em uma árvore de decisão,
    mas usando uma versão ponderada que leva em conta a importância diferenciada das classes através dos pesos
    """
    # Calcula entropia ponderada do conjunto original (antes da divisão)
    total_weighted_entropy = weighted_entropy(y, weights)

    # Calcula entropia ponderada média dos splits (depois da divisão)
    splits_entropy = sum([weighted_entropy(split, weights) * (len(split) / len(y)) for split in splits])

    # O ganho de informação é a diferença entre a entropia total e a média ponderada das entropias dos splits
    return total_weighted_entropy - splits_entropy


def mse_criterion(y, splits):
    y_mean = np.mean(y)
    return -sum([np.sum((split - y_mean) ** 2) * (float(split.shape[0]) / y.shape[0]) for split in splits])


def xgb_criterion(y, left, right, loss):
    left = loss.gain(left["actual"], left["y_pred"])
    right = loss.gain(right["actual"], right["y_pred"])
    initial = loss.gain(y["actual"], y["y_pred"])
    gain = left + right - initial
    return gain


def get_split_mask(X, column, value):
    left_mask = X[:, column] < value
    right_mask = X[:, column] >= value
    return left_mask, right_mask


def split(X, y, value):
    left_mask = X < value
    right_mask = X >= value
    return y[left_mask], y[right_mask]


def split_dataset(X, target, column, value, return_X=True):
    left_mask, right_mask = get_split_mask(X, column, value)

    left, right = {}, {}
    for key in target.keys():
        left[key] = target[key][left_mask]
        right[key] = target[key][right_mask]

    if return_X:
        left_X, right_X = X[left_mask], X[right_mask]
        return left_X, right_X, left, right
    else:
        return left, right


class BalancedTree(object):
    """Implementação recursiva da Decision Tree com equilíbrio entre classes."""

    def __init__(self, regression=False, criterion=None, n_classes=None):
        self.regression = regression
        self.impurity = None
        self.threshold = None
        self.column_index = None
        self.outcome = None
        self.criterion = criterion
        self.loss = None
        self.n_classes = n_classes
        self.left_child = None
        self.right_child = None
        # Armazenar estatísticas do nó para uso posterior
        self.node_stats = None

    @property
    def is_terminal(self):
        return not bool(self.left_child and self.right_child)

    def _find_splits(self, X):
        split_values = set()
        x_unique = list(np.unique(X))
        for i in range(1, len(x_unique)):
            average = (x_unique[i - 1] + x_unique[i]) / 2.0
            split_values.add(average)

        return list(split_values)

    def _find_best_split(self, X, target, n_features, weights):
        subset = random.sample(list(range(0, X.shape[1])), n_features)
        max_gain, max_col, max_val = None, None, None

        for column in subset:
            split_values = self._find_splits(X[:, column])
            for value in split_values:
                if self.loss is None:
                    # Random forest
                    splits = split(X[:, column], target["y"], value)
                    gain = self.criterion(target["y"], splits, weights)
                else:
                    # Gradient boosting
                    left, right = split_dataset(X, target, column, value, return_X=False)
                    gain = xgb_criterion(target, left, right, self.loss)

                if (max_gain is None) or (gain > max_gain):
                    max_col, max_val, max_gain = column, value, gain
        return max_col, max_val, max_gain

    def _train(self, X, target, max_features=None, min_samples_split=10, max_depth=None, minimum_gain=0.01,
               weights=None):
        try:
            assert X.shape[0] > min_samples_split
            assert max_depth > 0

            if max_features is None:
                max_features = X.shape[1]

            column, value, gain = self._find_best_split(X, target, max_features, weights)
            assert gain is not None
            if self.regression:
                assert gain != 0
            else:
                assert gain > minimum_gain

            self.column_index = column
            self.threshold = value
            self.impurity = gain

            # Split dataset
            left_X, right_X, left_target, right_target = split_dataset(X, target, column, value)

            # Grow left and right child
            self.left_child = BalancedTree(self.regression, self.criterion, self.n_classes)
            self.left_child._train(
                left_X, left_target, max_features, min_samples_split, max_depth - 1, minimum_gain, weights
            )

            self.right_child = BalancedTree(self.regression, self.criterion, self.n_classes)
            self.right_child._train(
                right_X, right_target, max_features, min_samples_split, max_depth - 1, minimum_gain, weights
            )
        except AssertionError:
            self._calculate_leaf_value(target, weights)

    def train(self, X, target, max_features=None, min_samples_split=10, max_depth=None, minimum_gain=0.01, loss=None):
        if not isinstance(target, dict):
            target = {"y": target}

        # Loss for gradient boosting
        if loss is not None:
            self.loss = loss

        if not self.regression:
            self.n_classes = len(np.unique(target['y']))

        # Usar pesos adaptativos para equilibrar as classes
        weights = adaptive_weights_calculate(target['y'])

        self._train(X, target, max_features=max_features, min_samples_split=min_samples_split,
                    max_depth=max_depth, minimum_gain=minimum_gain, weights=weights)

    def _calculate_leaf_value(self, targets, weights):
        if self.loss is not None:
            # Gradient boosting
            self.outcome = self.loss.approximate(targets["actual"], targets["y_pred"])
        else:
            # Random Forest
            if self.regression:
                # Mean value for regression task
                self.outcome = np.mean(targets["y"])
            else:
                # Probability for classification task with balanced weighting
                class_counts = np.bincount(targets["y"], minlength=self.n_classes)
                total_samples = targets["y"].shape[0]

                # Aplica os pesos
                weighted_counts = np.array([class_counts[i] * weights.get(i, 1) for i in range(self.n_classes)])

                # Normalização
                self.outcome = weighted_counts / np.sum(weighted_counts)

                # Armazena estatísticas do nó para uso posterior
                self.node_stats = {
                    'class_counts': class_counts, # Contagem por classe
                    'total_samples': total_samples, # Total de amostras no nó
                    'purity': np.max(class_counts) / total_samples if total_samples > 0 else 0 # Pureza do nó
                }

    def predict_row(self, row):
        """Prevê a classe de uma linha"""
        if not self.is_terminal:
            if row[self.column_index] < self.threshold:
                return self.left_child.predict_row(row)
            else:
                return self.right_child.predict_row(row)
        return self.outcome

    def predict_row_with_stats(self, row):
        """prevê a classe de uma linha e retorna estatísticas do nó
            estatisticas do nó permitem avaliar o quão confiável é a previsão
        """
        if not self.is_terminal:
            if row[self.column_index] < self.threshold:
                return self.left_child.predict_row_with_stats(row)
            else:
                return self.right_child.predict_row_with_stats(row)
        return self.outcome, self.node_stats

    def predict(self, X):
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            result[i] = self.predict_row(X[i, :])
        return result


class BaseEstimator:
    y_required = True
    fit_required = True

    def _setup_input(self, X, y=None):
        """Ensure inputs to an estimator are in the expected format."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError("Got an empty matrix.")

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError("Missed required argument y")

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError("The targets array must be no-empty.")

        self.y = y

    def fit(self, X, y=None):
        self._setup_input(X, y)

    def predict(self, X=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError("You must call `fit` before `predict`")

    def _predict(self, X=None):
        raise NotImplementedError()


class BalancedRandomForest(BaseEstimator):
    def __init__(self, n_estimators=200, max_features=None, min_samples_split=10, max_depth=None, criterion=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = []

    def fit(self, X, y):
        self._setup_input(X, y)
        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))
        else:
            assert X.shape[1] > self.max_features
        self._train()

    def _train(self):
        for tree in self.trees:
            tree.train(
                self.X,
                self.y,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )

    def _predict(self, X=None):
        raise NotImplementedError()


# Versão híbrida que combina árvores padrão e árvores balanceadas
class HybridRandomForestClassifier(BaseEstimator):
    def __init__(self, n_estimators=1000, max_features=None, min_samples_split=10, max_depth=1000,
                 criterion="entropy", balanced_ratio=0.5):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.balanced_ratio = balanced_ratio

        #lista de arvores balanceadas(usam balanced_information_gain) e padrao
        self.balanced_trees = []
        self.standard_trees = []

        self.minority_class_ = None
        self.majority_class_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        self._setup_input(X, y)

        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))
        else:
            assert X.shape[1] > self.max_features

        class_counts = np.bincount(y)
        self.minority_class_ = np.argmin(class_counts)
        self.majority_class_ = np.argmax(class_counts)
        self.n_classes_ = len(class_counts)

        n_balanced = int(self.n_estimators * self.balanced_ratio)
        n_standard = self.n_estimators - n_balanced

        self.balanced_trees = []
        for _ in range(n_balanced):
            tree = BalancedTree(criterion=balanced_information_gain)
            tree.train(
                self.X,
                self.y,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )
            self.balanced_trees.append(tree)


        self.standard_trees = []
        for _ in range(n_standard):
            def standard_information_gain(y, splits, _):
                return balanced_information_gain(y, splits, {i: 1 for i in range(self.n_classes_)})

            tree = BalancedTree(criterion=standard_information_gain)
            tree.train(
                self.X,
                self.y,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )
            self.standard_trees.append(tree)

        return self

    def predict(self, X=None):
        """Faz previsões combinando os resultados das árvores padrão e balanceadas"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X=None):
        """Retorna as probabilidades de classe combinando os resultados das árvores padrão e balanceadas"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))

        if self.balanced_trees:
            for tree in self.balanced_trees:
                for i, x in enumerate(X):
                    proba[i] += tree.predict_row(x)

        if self.standard_trees:
            for tree in self.standard_trees:
                for i, x in enumerate(X):
                    proba[i] += tree.predict_row(x)

        n_trees = len(self.balanced_trees) + len(self.standard_trees)
        proba /= n_trees

        return proba
