You need to get some files from the link below to the directory other_survey_data
link is : https://www.canfar.net/storage/vault/list/cirada/data/vlass_cat/

code to tidy up component table, produce host ID table, and finalise the catalogue

vlass_compcat_vlad_stage2_yg.py tidies up pybdsf+vlad output, adds flags, run as
*python3 vlass_compcat_vlad_stage2_yg.py vlad_filename subtile_filename nvss_filename first_filename*


host id files as currently setup to run separately (will be a single pipeline in future):

    1. vlass_iso_and_cd_finding.py selects candidate sources, run as
    *python3 vlass_iso_and_cd_finding.py component_filename*
    2. vlass_uw_lr_v2.py grabs unwise data on the fly, deleting once used, and runs likelihood_ratio_matching.py on each unwise data set, outputting matches, run as
    *python3 vlass_uw_lr_v2.py sourcelist_filename unwise_imagelist_filename*
    3. stack_matches.py merges the individual LR outputs into a single table keeping the most likely, run as
    *python3 stack_matches.py sourcelist_filename components_filename*

VLASSQL1CIR_finalise_cat.py adds flagging to Host ID table, updates the component table based on the Host ID table, and finalises the catalogue output. Run as
*python3 VLASSQL1CIR_finalise_cat.py components_filename hosts_filename subtile_filename*
