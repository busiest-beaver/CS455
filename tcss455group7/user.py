from lxml import etree
from copy import copy

'''
A data class for the user model.
'''
class user:

    __ATTR_NAMES = ['age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu']

    '''
    kwarg parameters:
        required: 'userid'
        optional: 'age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu'

    available attributes:
        id: the user id
        attrs: a hash map of user attributes
    '''
    def __init__(self, **kwargs):
        self.__attrs = {}
        assert kwargs['userid'] is not None
        self.__id = kwargs['userid']
        self.set(**kwargs)

    '''
    Get user id.
    '''
    def id(self, **kwargs):
        if 'id' in kwargs:
            self.__id = kwargs['id']
        return self.__id

    '''
    Get a copy of user attributes.
    '''
    def attrs(self):
        return copy(self.__attrs)

    '''
    Set attributes.
    '''
    def set(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__ATTR_NAMES:
                self.__attrs[k] = str(v)


    def to_xml(self):
        root = etree.Element('user')
        root.set('id', str(self.id()))
        root.set('age_group', str(self.__attrs.get('age', '-')))
        root.set('gender', str(self.__attrs.get('gender', '-')))
        root.set('extrovert', str(self.__attrs.get('ext', '-')))
        root.set('neurotic', str(self.__attrs.get('neu', '-')))
        root.set('agreeable', str(self.__attrs.get('agr', '-')))
        root.set('conscientious', str(self.__attrs.get('con', '-')))
        root.set('open', str(self.__attrs.get('ope', '-')))
        return etree.tostring(root, pretty_print=True)
