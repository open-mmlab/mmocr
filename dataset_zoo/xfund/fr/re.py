_base_ = ['ser.py']

_base_.train_preparer.parser.type = 'XFUNDREAnnParser'
_base_.train_preparer.packer.type = 'REPacker'
_base_.test_preparer.parser.type = 'XFUNDREAnnParser'
_base_.test_preparer.packer.type = 'REPacker'

config_generator = dict(type='REConfigGenerator')
