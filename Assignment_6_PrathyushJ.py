import json
import collections
import numpy
from Assignment_Five import NNData
from Assignment_Five import load_XOR
from Assignment_Five import FFBPNetwork


class MultiTypeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, collections.deque):
            return {"__deque__": list(o)}
        elif isinstance(o, numpy.ndarray):
            return {"__NDarray__": o.tolist()}
        elif isinstance(o, NNData):
            return {"__NNData__": o.__dict__}
        else:
            json.JSONEncoder.default(o)  # Hi Professor Reed, no matter how much I try editing my code, it always gives
            # me the same error of "TypeError: default() missing 1 required positional argument: 'o' which has been
            # restricting me from testing the rest of the program


def multi_type_decoder(o):
    if "__deque__" in o:
        return collections.deque(o["__deque__"])
    if "__NNData__" in o:
        dec_obj = o["__NNData__"]
        features = dec_obj["_features"]
        labels = dec_obj["_labels"]
        train_indices = dec_obj["_train_indices"]
        test_indices = list(dec_obj["_test_indices"])
        train_pool = list(dec_obj["_train_pool"])
        test_pool = list(dec_obj["_test_pool"])
        train_factor = dec_obj["_train_factor"]
        ret_obj = NNData(features, labels, train_indices, test_indices, train_pool, test_pool, train_factor)
        return ret_obj
    if "__NDarray__" in o:
        return numpy.array(o["__NDarray__"])
    else:
        return o


with open("dat.txt", "w") as f:
    json.dump(NNData, f, cls=MultiTypeEncoder)

with open("dat.txt", "r") as f:
    my_obj = json.load(f, object_hook=multi_type_decoder)
    print(type(my_obj))
    print(my_obj)


def main():
    xor_data = NNData(load_XOR())
    with open("dat.txt", "w") as xor_data_encoded:
        json.dump(xor_data, xor_data_encoded, cls=MultiTypeEncoder)
    xor_data_decoded = NNData()
    with open("dat.txt", "w") as xor_data_decoded:
        json.dump(xor_data_encoded, xor_data_decoded, cls=multi_type_decoder())
    with open("sin_data.txt", "w") as xor_data_decoded:
        json.dump(xor_data_encoded, xor_data_decoded, cls=multi_type_decoder())


if __name__ == "__main__":
    main()
