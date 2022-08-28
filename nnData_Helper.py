
class DataHelper():
    def is_list_of_lists(container):
        if not isinstance(container, list):
            raise TypeError('Container must be of data type: List.')
            return False
        else:
            for idx, item in enumerate(container):
                if not isinstance(item, list):
                    msg = 'Items in container must be of data type "list"'
                    msg += f'Error item: index: {idx}.'
                    raise TypeError(msg)
                    return False
        return True
