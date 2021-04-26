def clean(txt):
    txt = txt.lower()
    lixo = [",", "-", "foxbot", "."]
    for s in lixo:
        txt = txt.replace(s, "")
    return txt