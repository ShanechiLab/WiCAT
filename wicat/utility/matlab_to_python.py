from __future__ import annotations

from typing import Any

import scipy.io as spio

__all__ = ["loadmat"]


def _tolist(value: Any):
	if getattr(value, "dtype", None) == object:
		return [_todict(v) if _ismatstruct(v) else _tolist(v) for v in value]
	return value


def _ismatstruct(value: Any) -> bool:
	return isinstance(value, spio.matlab.mio5_params.mat_struct)


def _todict(matobj: Any):
	output = {}
	for field_name in matobj._fieldnames:
		elem = getattr(matobj, field_name)
		if _ismatstruct(elem):
			output[field_name] = _todict(elem)
		elif getattr(elem, "dtype", None) == object:
			output[field_name] = _tolist(elem)
		else:
			output[field_name] = elem
	return output


def _check_keys(data: dict):
	for key in list(data.keys()):
		value = data[key]
		if _ismatstruct(value):
			data[key] = _todict(value)
		elif getattr(value, "dtype", None) == object:
			data[key] = _tolist(value)
	return data


def loadmat(path: str, **kwargs):
	default_kwargs = dict(struct_as_record=False, squeeze_me=True)
	default_kwargs.update(kwargs)
	data = spio.loadmat(path, **default_kwargs)
	return _check_keys(data)
