data_root = 'data/sroie'
cache_path = 'data/cache'

data_obtainer = dict(
 type='NaiveDataObtainer',
 cache_path=cache_path,
 data_root=data_root,
 files=[
        dict(
            url='https://doc-b4-c8-drive-data-export.googleusercontent.com/download/'
            'fjpfehrefh7sl37sr71jeh2bhss0bjnc/ipsc6m9rhnufrgfou2v150pf2qvu85jj/1671538500000'
            '/ab5eb9ff-0939-4733-b13e-b91ccd5ca3e9/103212024722882466354/ADt3v-PlZRXqj94_z156'
            'zbyAJvCgP_INPDXqzUqez3jcQBO0TCy1AXXkuWPgRLCbyQC5x7cisBfEtqGdAHlGKo9yJnvAcv12GsAHS'
            '2ppinZhj2_kFlcQqmCIUogMe2iKzsSZ3RMSbgTMIfIMSlDcZ-UTI5OfXCLR3JhZ31ZCxYBYEmrqaMSrO_L5'
            '5dUzNw-pgCxeE56qpoLYWQo-oGmpKfX4fLdbJkNxcC49uXCHHdHifHjpTCFn0B6-XCVdp9uX0fweojHWNiy'
            'D_Z7WRzIOmrBF0MM3tmlCN8O4Txu8gskuUPqvc66WzRaIFXcfEdYq_QLe4gD-8fr8L6q4-u26TsDAly2f9937'
            'mA==?authuser=0&nonce=vu16mp02amtcg&user=103212024722882466354&hash=bgb1v1uca3q9sik1b'
            'crt70u7drv202cu',
            save_name='0325updated.task1train(626p)-20221205T094142Z-001.zip',
            md5='3883fa9ef6ade95a8d3b9076813ad848',
            split=['train'],
            content=['image', 'annotation'],
            mapping=[['0325updated/0325updated.task1train(626p)/*.jpg', 'textdet_imgs/train'],
                     ['0325updated/0325updated.task1train(626p)/*.txt', 'annotations/train']]),
        dict(
            url='https://doc-0o-2o-drive-data-export.googleusercontent.com/download/'
            'fjpfehrefh7sl37sr71jeh2bhss0bjnc/16u970fcc7dd54arckvrpjbd6ecdkull/1671538500000/'
            'b3194449-f672-4f77-8021-1a9650df6aa4/103212024722882466354/ADt3v-MXYOvylQYausMHi'
            'x7WhseI1d0YLyqtB2r8n1QPTJebve8oJFf4_8APh0L7r-HyE0CyjhxRXf8bM909oFE2sWKEulqEnvussV'
            'QdvFh73dlC7goMbGSb1-EfUWR4wXpHpsoPYVOjw1grQiExl0v_P3LHaD9ErTkZkA3ZhR-q9iKUq_7i7eyX'
            'bdhT35l34Xnal7mFap2P2ZdacoBLzD2LDI1GRqXtlnMKWm4KnTZuBGBNXFxJGjjnhy3x4j9meSt_eod0v'
            'ZzyfAgs5ThHM1kk6dc8pTKAM84p0z2cp-N0GJgi8pLMZR8nPdxKBJNipOzp4Y_8tUGO?authuser=0&non'
            'ce=vkamrsalrga2q&user=103212024722882466354&hash=8trg3mjsqhc8hf6uhim8jqirsjdp0r2l',
            save_name='task1&2_test(361p)-20221205T104647Z-001.zip',
            md5='1b05af23e4b38ca27d19cfe95272418f',
            split=['test'],
            content=['image'],
            mapping=[['task1&2_test(361p)-20221205T104647Z-001/task1_2_test(361p)', 'textdet_imgs/test']]),
        dict(
            url='https://doc-4c-2o-drive-data-export.googleusercontent.com/download/'
            'fjpfehrefh7sl37sr71jeh2bhss0bjnc/70q53fnetl2bprm7931dubh782li4d3b/1671538500000/'
            '8c3588ab-67fe-4d4d-8a0c-d105e4cbcfd3/103212024722882466354/ADt3v-Ovr_nucmMNIjxpVb'
            'CHhn2p5_N7rtbldYPUtHF_k1dtcWih0LnE1BSUuHNhlf7MF4mcABQFhpaEIZX_GIkaDOUJ3IT8gSnMiz'
            '5aaK225clwIWyEBmZHqKC3e87Gz785sWQUSRLxucU3k2JCvyfI0uwnlozNcICYfRn2pPnCmGL4-iqm6Mts'
            'H9fzy5p_3nWpgw6TN5q14-2wT4CsIdMK_kPizJCzkVGkYkiRmYp3AfQso1kxGn5x1h1KsI9ofz9VnJJDmL'
            'ITqILz7ax8_5IIsMVkgDkxgFJM8pUfdVPQw2Y1n96Hm3M0UGWM2m-ZVxQJExRLTDL-CCC1AWkLAY6EwWFmpn'
            'Dtdg==?authuser=0&nonce=tiu0m27r7pvvm&user=103212024722882466354&hash=6uq9tev1b1c9q'
            '3gn83qsifbl42klefkm',
            save_name='text.task1&2-test（361p)-20221205T112052Z-001.zip',
            md5='0bf94b74a5baadfe2bb0cd72889127a5',
            split=['test'],
            content=['annotation'],
            mapping=[['text/text.task1_2-test（361p)', 'annotations/test']]),
    ])


data_converter = dict(
    type='TextRecogCropConverter',
    splits=['train', 'test'],
    data_root=data_root,
    gatherer=dict(
        type='pair_gather',
        suffixes=['.jpg'],
        rule=[r'X(\d+)\.([jJ][pP][gG])', r'X\1.txt']),
    parser=dict(type='SROIETextDetAnnParser', encoding='utf-8-sig'),
    dumper=dict(type='JsonDumper'),
    delete=['text', 'task1&2_test(361p)-20221205T104647Z-001', '0325updated', 'annotations']
)

config_generator = dict(type='TextRecogConfigGenerator', data_root=data_root)
