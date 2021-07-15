  
import time
import os
import streamlit as st
import SessionState
import numpy as np
from PIL import Image
import pandas as pd
state = SessionState.get(result_text="", res="", prob_positive=0.0, prob_negative= 0.0, initial=True, img_drawed=None, img_cropped=None, reg_text_time=None)


def main():
    
    st.title("DMEC Demo")
    # Load model

    pages = {
        'DMEC check': page_DMEC

    }

    st.sidebar.title("Application")
    page = st.sidebar.radio("Demo application:", tuple(pages.keys()))
    
    # st.title('Model Options')
    st.sidebar.title('TBYT')
    info = st.sidebar.file_uploader('Upload TBYT file')

    # st.sidebar.title('Submit time')
    tim = st.sidebar.text_input('Submit time')

    st.sidebar.subheader('Authorization letter')
    auth_letter = st.sidebar.file_uploader('Upload authorization letter')

    # print(auth_letter)
    if auth_letter is not None:
        with open(auth_letter.name,"wb") as f:
            f.write(auth_letter.getbuffer())
        
    st.sidebar.subheader('ISO certificate')
    iso = st.sidebar.file_uploader('Upload ISO certificate')

    st.sidebar.subheader('CFS certificate')
    cfs = st.sidebar.file_uploader('Upload CFS certificate')

    st.sidebar.subheader('Equipment classification letter')
    eqmt = st.sidebar.file_uploader('Upload equipment classification letter')

    # model = load_model(latency, card_model, line_model, text_model)
    model = None

    pages[page](state, model)

    # state.sync()


@st.cache(allow_output_mutation=True)  # hash_func
def load_model(latency, card_model, line_model, text_model):
    # print("Loading model ...")
    # model = TEXT_IMAGES(latency=latency, line_model=line_model, card_model=card_model, text_model=text_model, reg_model='vgg_seq2seq', ocr_weight_path='weights/seq2seqocr_best.pth')
    model=None
    return model


def page_DMEC(state, model):
    st.header("RESULTS")
    
    # result = model()
    result = {
    "profileId": 0,
    "profileCode": "string",
    "status": "OK",
    "result": {
        "classification": {
            "code": "string",
            "comments": [{
                "comment": "Tên thiết bị chính xác với thông tin Đăng ký lưu hành (trang 2)",
                "status": "OK"
            }, {
                "comment": "Loại thiết bị chính xác với thông tin Đăng ký lưu hành (trang 2)",
                "status": "OK"
            }, {
                "comment": "Chủng loại/ mã sản phẩm chính xác với thông tin Đăng ký lưu hành (trang 2)",
                "status": "OK"
            }, {
                "comment": "Hãng, nước sản xuất chính xác với thông tin Đăng ký lưu hành (trang 2)",
                "status": "OK"
            }, {
                "comment": "Hãng, nước chủ sở hữu chính xác với thông tin Đăng ký lưu hành (trang 2)",
                "status": "OK"
            }]
        },
        "iso": {
            "code": "string",
            "comments": [{
                "id": 0,
                "status": "OK",
                "comments": [{
                    "comment": " iso 13485 chính xác (trang 1)",
                    "status": "OK"
                }, {
                    "comment": " is hết thời gian ",
                    "status": "NOT OK"
                }, {
                    "comment": " Tên cơ sở sản xuất chính xác (trang 2)",
                    "status": "OK"
                }, {
                    "comment": " Địa chỉ cơ sở sản xuất chính xác (trang 2)",
                    "status": "OK"
                }]
            }]
        },
        "attorneyPower": {
            "code": "string",
            "comments": [{
                "id": 0,
                "status": "OK",
                "comments": [{
                    "comment": "Có dấu lãnh sứ quán (trang 5)",
                    "status": "OK"
                }, {
                    "comment": "Địa chỉ công ty sở hữu hợp lệ (trang 1)",
                    "status": "OK"
                }, {
                    "comment": "Địa chỉ công ty được ủy quyền hợp lệ (trang 2)",
                    "status": "OK"
                }, {
                    "comment": "Tên công ty được ủy quyền hợp lệ (trang 2)",
                    "status": "OK"
                }, {
                    "comment": "Tên công ty sở  hữu hợp lệ (trang 2)",
                    "status": "OK"
                }, {
                    "comment": "Thời gian hiệu lực phù hợp",
                    "status": "OK"
                }, {
                    "comment": "Danh sách TTBYT đầy đủ",
                    "status": "OK"
                }]
            }]
        },
        "cfsLocal": {
            "code": "string",
            "comments": []
        },
        "cfsForeign": {
            "code": "string",
            "comments": [{
                "comment": "Công ty sản xuất hợp lệ (trang 1)",
                "status": "OK"
            }, {
                "comment": "Công ty sở hữu hợp lệ (trang 1)",
                "status": "OK"
            }, {
                "comment": "Danh sách TTBYT đầy đủ",
                "status": "OK"
            }, {
                "comment": "Có dấu lãnh sứ quán (trang 2)",
                "status": "OK"
            }]
        }
    }
}

    cls_result = reconstruct(result['result']['classification']['comments'])
    iso_result = reconstruct(result['result']['iso']['comments'][0]['comments'])
    att_result = reconstruct(result['result']['attorneyPower']['comments'][0]['comments'])
    cfs_result = reconstruct(result['result']['cfsLocal']['comments'] if len(result['result']['cfsLocal']['comments'])>0 else result['result']['cfsForeign']['comments'])
    
    cls_result = pd.DataFrame(cls_result)
    iso_result = pd.DataFrame(iso_result)
    att_result = pd.DataFrame(att_result)
    cfs_result = pd.DataFrame(cfs_result)

    cls_result = cls_result.style.apply(highlight_rows, axis=1)
    iso_result = iso_result.style.apply(highlight_rows, axis=1)
    att_result = att_result.style.apply(highlight_rows, axis=1)
    cfs_result = cfs_result.style.apply(highlight_rows, axis=1)

    # col1, col2 = st.beta_columns(2)
    # with col1:\
    st.title('Classification')
    st.table(cls_result)
    st.title('ISO')
    st.table(iso_result)
    
    # with col2:
    st.title('AttorneyPower')
    st.table(att_result)
    st.title('CFS')
    st.table(cfs_result)

    
def reconstruct(result):
    construct = []
    for res in result:
        dat = {}
        keys = res.keys()
        for key in keys:
            if key == 'status':
                dat['status'] = res[key]
            else:
                # dat['checklist'] = key
                dat['comment'] = res[key]
        construct.append(dat)
    return construct

def highlight_rows(df):
      
    # overwrite values grey color
    if df['status'] == 'OK':
        return ['']*2
    
    else:
        return ['background-color: red']*2 

if __name__ == "__main__":
    main()
