import cv2
import numpy as np

def main():
    # 1. 이미 저장된 원본 크롭과 렌더링 결과 불러오기
    query_path = "data/query_ds/q2_00044_ds.png"
    render_path = "data/outputs_0417/q2_00044_3dgs_scale_ds_100iter/refine_pose_gs_ds/refined_render_full.png"
    output_path = "data/outputs_0417/q2_00044_3dgs_scale_ds_100iter/refine_pose_gs_ds"

    query = cv2.imread(query_path)
    render = cv2.imread(render_path)
    
    if query is None or render is None:
        print("이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # 2. 이미지 해상도가 같은지 확인
    if query.shape != render.shape:
        print(f"해상도가 다릅니다! Query: {query.shape}, Render: {render.shape}")
        return

    # 3. Overlay (알파 블렌딩) 만들기
    alpha = 0.5  # 0.5로 주면 원본 50%, 렌더링 50%로 겹쳐 보임
    overlay = cv2.addWeighted(query, alpha, render, 1.0 - alpha, 0)
    
    # 4. 비교하기 쉽게 3장을 나란히 붙이기 (가로로 병합)
    # [ 원본(Query) | 렌더링(Render) | 겹친 화면(Overlay) ]
    comp = np.hstack((query, render, overlay))
    
    # 글자(라벨) 넣기
    h, w, _ = query.shape
    cv2.putText(comp, "Query", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(comp, "Render", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(comp, "Overlay (50%)", (w * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 5. 결과 저장
    out_path_3 = f"{output_path}/overlay_check_00044.png"
    out_path_overlay = f"{output_path}/query-refine overlay_00044.png"
    cv2.imwrite(out_path_3, comp)
    cv2.imwrite(out_path_overlay, overlay)
    print(f"완료! '{out_path_3}', '{out_path_overlay}' 파일이 생성되었습니다.")

if __name__ == "__main__":
    main()