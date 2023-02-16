def money_entity(token):
    keywords = ['שולם','שילם', 'כסף', 'דולר']
    if token in keywords:
        return 'I-Money'
    return 'O'