import configparser


def ini_conf_file(wavelet):

    if wavelet == "niko":
        kernel = "morlet"
    elif wavelet == "pywt":
        kernel = "cmor1.5-1.0"
    else:
        raise Exception("wrong kernel name!!! GIVE ME GOOD NAME")

    data_folder = "./data/output/"
    input_folder = "ENSO_manuel_" + wavelet + "_nSka_40/"
    export_folder = "TE_ENSO_manuel_month_" + wavelet + "-p6-84mth/"

    in_data_tag = "ENSO_manuel_" + kernel + "_lin-p6-84mth"
    sufixes = "_pha.npy,_amp.npy,_ska.npy"
    pha_amp_com = "_pha,_pha"
    pha_amp_more = "_pha,_amp"

    # CREATE OBJECT
    config_file = configparser.ConfigParser()

    ###ADD SECTION###

    config_file.add_section("folders")

    # section settings
    config_file.set("folders", "data_folder", data_folder)
    config_file.set("folders", "input_folder", input_folder)
    config_file.set("folders", "export_folder", export_folder)

    ###ADD SECTION###
    config_file.add_section("names")

    # section settings
    config_file.set("names", "in_data_tag", in_data_tag)
    config_file.set("names", "sufixes", sufixes)
    config_file.set("names", "pha_amp_com", pha_amp_com)
    config_file.set("names", "pha_amp_more", pha_amp_more)

    ###ADD SECTION###
    config_file.add_section("emb_par")

    # section settings
    config_file.set("emb_par", "bins", "150")
    config_file.set("emb_par", "max_tau", "10")
    config_file.set("emb_par", "jump_tau", "5")
    config_file.set("emb_par", "embeding_dimension", "2,1,1,1,1")
    config_file.set("emb_par", "period_range", "6,84")

    ###ADD SECTION###
    config_file.add_section("prob_est")

    # section settings
    config_file.set("prob_est", "name_tag", "Prob-est_VisFreq_b")
    config_file.set("prob_est", "prob_kind", "VisFreq")

    ###ADD SECTION###
    config_file.add_section("surrogates")

    # section settings
    config_file.set("surrogates", "surr_kind", "circular")
    config_file.set("surrogates", "surr_num", "11")

    return config_file


def save_conf_file(name, config):
    # SAVE CONFIG FILE

    with open(name, "w") as configfile:  # save
        config.write(configfile)

    print("Config file", name, " created")


def load_config(name):

    # PRINT FILE CONTENT
    config_file = configparser.ConfigParser()
    config_file.read(name)
    # read_file = open(name, "r")
    # content = read_file.read()
    print("Content of the config file are:\n")
    print(config_file["names"]["sufixes"], "\n")
    # print(content)
    # read_file.flush()
    # read_file.close()

    return config_file


def main():
    wavelet = "niko"
    # wavelet = 'pywt'
    config = ini_conf_file(wavelet)
    name = "./confs/conf_embeding_char.ini"
    save_conf_file(name, config)
    load_config(name)


if __name__ == "__main__":
    main()
