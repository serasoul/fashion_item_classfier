# 이하를 「app.py」에 써넣기
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("이미지 인식 앱")
st.sidebar.write("오리지널 이미지 인식 모델을 사용해서 무슨 이미지인지를 판정합니다.")

st.sidebar.write("")

img_source = st.sidebar.radio("이미지 소스를 선택해 주세요.",
                              ("이미지를 업로드", "카메라로 촬영"))
if img_source == "이미지를 업로드":
    img_file = st.sidebar.file_uploader("이미지를 선택해 주세요.", type=["png", "jpg", "jpeg"])
elif img_source == "카메라로 촬영":
    img_file = st.camera_input("카메라로 촬영")

if img_file is not None:
    with st.spinner("측정 중..."):
        img = Image.open(img_file)
        st.image(img, caption="대상 이미지", width=480)
        st.write("")

        # 예측
        results = predict(img)

        # 결과 표시
        st.subheader("판정 결과")
        n_top = 3  # 확률이 높은 순으로 3위까지 반환한다
        for result in results[:n_top]:
            st.write(str(round(result[2]*100, 2)) + "%의 확률로 " + result[0] + "입니다.")

        # 원그래프 표시
        pie_labels = [result[1] for result in results[:n_top]]
        pie_labels.append("others")  # 기타
        pie_probs = [result[2] for result in results[:n_top]]
        pie_probs.append(sum([result[2] for result in results[n_top:]]))  # 기타
        fig, ax = plt.subplots()
        wedgeprops={"width":0.3, "edgecolor":"white"}
        textprops = {"fontsize":6}
        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
               textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)  # 원그래프
        st.pyplot(fig)

st.sidebar.write("")
st.sidebar.write("")

st.sidebar.caption("""
이 앱은「Fashion-MNIST」를 훈련 데이터로 사용하고 있습니다.\n
Copyright (c) 2017 Zalando SE\n
Released under the MIT license\n
https://github.com/zalandoresearch/fashion-mnist#license
""")
