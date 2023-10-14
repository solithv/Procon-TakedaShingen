from Utils import API


def main():
    fa = API()
    match = fa.get_match()
    if len(match) != 1:
        print("match is not one")
    match = match[0]
    id_ = match["id"]

    print("id:", id_)


if __name__ == "__main__":
    main()
