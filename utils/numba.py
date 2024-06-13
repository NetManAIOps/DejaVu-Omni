import numba as nb

__all__ = [
    "py_object_to_nb_object",
]

from pyprof import profile


@profile
def py_object_to_nb_object(py_obj):
    """
    Convert a Python object to a Numba object.
    Credit to https://github.com/numba/numba/issues/4728
    :param py_obj:
    :return:
    """
    if type(py_obj) == dict:
        py_dict = py_obj
        keys = list(py_dict.keys())
        values = list(py_dict.values())
        if type(keys[0]) == str:
            nb_dict_key_type = nb.types.string
        elif type(keys[0]) == int:
            nb_dict_key_type = nb.types.int64
        elif type(keys[0]) == float:
            nb_dict_key_type = nb.types.float64
        else:
            raise TypeError(f"Dictionary keys must be strings, ints, or floats, got {type(keys[0])}")
        if type(values[0]) in {int, str, float}:
            if type(values[0]) == str:
                nb_dict_value_type = nb.types.string
            elif type(values[0]) == int:
                nb_dict_value_type = nb.types.int64
            elif type(values[0]) == float:
                nb_dict_value_type = nb.types.float64
            else:
                raise TypeError(f"{type(values[0])=} is not int, str or float")
            nbh = nb.typed.Dict.empty(nb_dict_key_type, nb_dict_value_type)
            for i, key in enumerate(keys):
                nbh[key] = values[i]
            return nbh
        elif type(values[0]) == dict:
            for i, subDict in enumerate(values):
                subDict = py_object_to_nb_object(subDict)
                if i == 0:
                    nbh = nb.typed.Dict.empty(nb_dict_key_type, nb.typeof(subDict))
                # noinspection PyUnboundLocalVariable
                nbh[keys[i]] = subDict
            return nbh
        elif type(values[0]) == list:
            for i, subList in enumerate(values):
                subList = py_object_to_nb_object(subList)
                if i == 0:
                    nbh = nb.typed.Dict.empty(nb_dict_key_type, nb.typeof(subList))
                # noinspection PyUnboundLocalVariable
                nbh[keys[i]] = subList
            return nbh
        else:
            raise TypeError(f"Dictionary values must be int, str, float, dict, or list, got {type(values[0])=}")
    elif type(py_obj) == list:
        pyList = py_obj
        data = pyList[0]
        if type(data) == int:
            nbs = nb.typed.List.empty_list(nb.types.int64)
            for data_ in pyList:
                nbs.append(data_)
            return nbs
        elif type(data) == str:
            nbs = nb.typed.List.empty_list(nb.types.string)
            for data_ in pyList:
                nbs.append(data_)
            return nbs
        elif type(data) == float:
            nbs = nb.typed.List.empty_list(nb.types.float64)
            for data_ in pyList:
                nbs.append(data_)
            return nbs
        elif type(data) == dict:
            for i, subDict in enumerate(pyList):
                subDict = py_object_to_nb_object(subDict)
                if i == 0:
                    nbs = nb.typed.List.empty_list(nb.typeof(subDict))
                # noinspection PyUnboundLocalVariable
                nbs.append(subDict)
            return nbs
        elif type(data) == list:
            for i, subList in enumerate(pyList):
                subList = py_object_to_nb_object(subList)
                if i == 0:
                    nbs = nb.typed.List.empty_list(nb.typeof(subList))
                # noinspection PyUnboundLocalVariable
                nbs.append(subList)
            return nbs
        else:
            raise TypeError(f"List values must be int, str, float, dict, or list, got {type(data)=}")
    else:
        raise TypeError("Unsupported type: {}".format(type(py_obj)))
