# HẢI ĐỒ SỐ — ỨNG DỤNG PHÂN TÍCH MẠNG VẬN TẢI HÀNG HẢI VIỆT NAM & QUỐC TẾ
> **Đồ án môn học: Phân tích Mạng Xã hội (Social Network Analysis - SNA)**  
> **Chương trình Thạc sĩ Công nghệ Thông tin**  
> **Giao diện: Chuẩn Nautical Light Mode — Sạch sẽ, Tinh tế, Hiện đại, Chuẩn mực Quốc tế**

---

## 1. Giới thiệu Tổng quan (Overview)

**Hải đồ số (Maritime SNA Explorer)** là ứng dụng web tương tác đa trang xây dựng trên nền tảng **Streamlit**, **Pydeck / Deck.gl (WebGL)**, **Plotly**, **NetworkX**, và **python-igraph**. Ứng dụng mô hình hóa toàn diện mạng lưới vận tải hàng hải toàn cầu với trọng tâm là hệ thống cảng biển và tuyến kết nối thương mại của Việt Nam từ hơn **1,157,000 bản ghi dữ liệu thực tế** (IMF PortWatch, UNCTAD, Global Fishing Watch, OpenSeaMap).

### Các điểm nhấn công nghệ & phương pháp luận:
- **Xử lý dữ liệu lớn & Tối ưu hoá Cold-Start**: Pipeline ngoại tuyến `build_artifacts.py` tính toán sẵn 16 tệp dữ liệu Parquet/JSON tối ưu (tổng dung lượng ~10.2 MB). Ứng dụng nạp dữ liệu tức thì trong **0.39 giây** và hoạt động độc lập offline.
- **Hệ thống thiết kế Nautical Light Mode**: Bảng màu hải quân hiện đại (#FFFFFF nền sáng, #F8FAFC thẻ card, #0077B6 xanh đại dương, #DA251D cờ đỏ Việt Nam, #F59E0B hổ phách), font chữ Be Vietnam Pro và số liệu JetBrains Mono tương phản cao.
- **Trực quan hoá WebGL không phụ thuộc Mapbox Token**: Tích hợp Carto Positron vector tiles hoàn toàn miễn phí và ổn định, kết hợp thành phần WebGL hoạt ảnh luồng tàu chạy **TripsLayer** có thanh tua thời gian và nút phát mượt mà.
- **Toán học & SNA Chuẩn mực**: 6 thuật toán phát hiện cộng đồng (Leiden, Louvain, LPA, Infomap, Greedy, Girvan-Newman), 6 mô hình dự đoán liên kết (Link Prediction ROC/PR), mô phỏng đứt gãy mạng (Percolation / Cascading Failures), bộ lọc xương sống Serrano Disparity Filter, và phân tích lỗ hổng cấu trúc Ronald Burt.

---

## 2. Cấu trúc Dự án (Repository Structure)

```
haido-sna/
├── app.py                         # App Shell chính đa trang (sử dụng st.navigation)
├── build_artifacts.py             # Pipeline tiền xử lý & tính toán offline 16 artifacts
├── requirements.txt               # Danh mục thư viện phụ thuộc Python
├── pytest.ini                     # Cấu hình kiểm thử tự động pytest
├── README.md                      # Tài liệu hướng dẫn cài đặt & vận hành
├── .streamlit/
│   └── config.toml                # Cấu hình giao diện Streamlit (Light theme, port, cache)
├── artifacts/                     # 16 tệp Parquet/JSON đã được tính toán sẵn
│   ├── nodes.parquet              # 2,424 cảng biển toàn cầu & 31 chỉ số SNA
│   ├── edges.parquet              # 169,888 tuyến vận tải biển & trọng số
│   ├── edges_geo.parquet          # Tuyến toạ độ bathymetric searoutes thực tế
│   ├── trips.json                 # 400 hành trình tàu chạy phục vụ TripsLayer
│   ├── communities.parquet        # Kết quả phân cụm 6 thuật toán cộng đồng
│   ├── community_eval.parquet     # Ma trận so sánh NMI, ARI và Modularity Q
│   ├── linkpred_eval.parquet      # Đánh giá AUC-ROC, AP, P@100 của 6 mô hình
│   ├── linkpred_top.parquet       # Top 10 tuyến kết nối tiềm năng cho 259 cảng VN
│   ├── robustness_baseline.parquet# Dữ liệu mô phỏng đứt gãy mạng & tấn công
│   ├── structure_stats.json       # Thống kê Small-World, Scale-Free, Rich-Club
│   ├── lsbci_long.parquet         # Chuỗi thời gian LSBCI UNCTAD 21 quý (2006-2026)
│   ├── lsbci_backbone.parquet     # Khung xương thương mại Serrano Disparity Filter
│   ├── daily_vn.parquet           # Dữ liệu lượt tàu ghé 18 cảng VN (5 loại tàu)
│   ├── ferry_nodes.parquet        # 25 bến phà biển & duyên hải Việt Nam
│   ├── ferry_edges.parquet        # 17 cặp tuyến phà kết nối nội địa
│   └── manifest.json              # Dấu thời gian, số bản ghi & kiểm tra tính toàn vẹn
├── core/                          # Thư viện thuật toán và quản lý dữ liệu lõi
│   ├── algorithms.py              # 100% thuật toán SNA chân thực (không mock/facade)
│   ├── data.py                    # Trình nạp dữ liệu đệm cache (@st.cache_data)
│   ├── graph.py                   # Thao tác đồ thị NetworkX, igraph, Yen k-paths
│   ├── seaway.py                  # Hình học hải trình bathymetric searoute
│   └── theme.py                   # Bảng màu Nautical Light Mode & ECDIS formatters
├── components/                    # Cấu phần giao diện tái sử dụng
│   ├── hud.py                     # Thẻ chỉ số HUD JetBrains Mono responsive
│   ├── layers.py                  # Nhà máy tạo layer Pydeck/Deck.gl đa tầng
│   ├── deck_map.py                # Wrapper bản đồ Pydeck Carto Positron
│   ├── animated_map.py            # Component nhúng TripsLayer hoạt ảnh tàu chạy
│   └── animated_map.html          # HTML/JS WebGL TripsLayer độc lập
├── pages/                         # 10 màn hình chức năng chuyên sâu
│   ├── 00_hai_do_tong_quan.py     # 00. Hải đồ tổng quan mạng cảng toàn cầu
│   ├── 01_ho_so_cang.py           # 01. Hồ sơ chi tiết cảng, Radar & Ego-Network
│   ├── 02_tim_tuyen.py            # 02. Tìm k tuyến đường biển & tránh điểm nghẽn
│   ├── 03_cong_dong.py            # 03. 6 thuật toán phát hiện cộng đồng mạng
│   ├── 04_mo_phong_su_co.py       # 04. Mô phỏng đóng eo biển & gián đoạn mạng
│   ├── 05_du_doan_tuyen_moi.py    # 05. Dự đoán tuyến mới cho cảng Việt Nam
│   ├── 06_viet_nam_toan_cau.py    # 06. Chuỗi thời gian LSBCI & Serrano Backbone
│   ├── 07_pha_noi_dia.py          # 07. Mạng phà nội địa & tìm đường di chuyển
│   ├── 08_cau_truc_mang.py        # 08. Cấu trúc vĩ mô, Small-World & Scale-Free
│   └── 09_phuong_phap.py          # 09. Phương pháp luận, Lineage & Trích dẫn
└── tests/                         # Bộ kiểm thử tự động toàn diện (99 tests)
    ├── conftest.py                # Fixtures kiểm thử & môi trường cô lập
    ├── test_data_contracts.py     # 30 bài kiểm thử hợp đồng dữ liệu 16 artifacts
    ├── test_graph.py              # 20 bài kiểm thử tô-pô đồ thị & thuật toán
    ├── test_algorithms.py         # 28 bài kiểm thử bất biến toán học SNA
    └── test_e2e_app.py            # 21 bài kiểm thử E2E giao diện & Headless AppTest
```

---

## 3. Hướng dẫn Cài đặt & Vận hành (Installation & Quick Start)

### Yêu cầu hệ thống:
- Hệ điều hành: Windows 10/11, macOS, hoặc Linux
- Phiên bản Python: **Python 3.10, 3.11, 3.12, hoặc 3.14**
- Trình duyệt web hiện đại hỗ trợ WebGL (Google Chrome, Microsoft Edge, Mozilla Firefox)

### Bước 1: Cài đặt các gói thư viện phụ thuộc
Mở Command Prompt hoặc PowerShell tại thư mục dự án `haido-sna/` và chạy lệnh:
```powershell
pip install -r requirements.txt
```
*Các thư viện chính bao gồm: `streamlit>=1.40.0`, `pydeck>=0.9.0`, `networkx>=3.2`, `igraph>=0.11.0`, `pandas>=2.0.0`, `numpy>=1.24.0`, `plotly>=5.18.0`, `scikit-learn>=1.3.0`, `powerlaw>=1.5`, `pyarrow>=14.0.0`, `pytest>=8.0.0`.*

### Bước 2 (Tuỳ chọn): Tái xây dựng 16 Artifacts từ dữ liệu gốc
*Lưu ý: Thư mục `artifacts/` đã có sẵn đầy đủ 16 tệp dữ liệu chuẩn hoá. Nếu muốn chạy lại toàn bộ pipeline tiền xử lý và tính toán toán học từ tệp nén gốc:*
```powershell
python build_artifacts.py
```
*Pipeline sẽ tự động đọc 1.157M bản ghi thô từ `hanghai_sna_output.zip`, tính toán các độ đo trung tâm, chạy 6 thuật toán cộng đồng, tìm kiếm đường biển qua searoute, mô phỏng đứt gãy mạng và xuất ra các tệp `.parquet` và `.json` trong thư mục `artifacts/`.*

### Bước 3: Khởi chạy Ứng dụng Web
Thực thi lệnh Streamlit:
```powershell
streamlit run app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: **`http://localhost:8501`**

---

## 4. Chi tiết 10 Màn hình Chức năng (Functional Modules)

### ⚓ Trang 00: Hải đồ Tổng quan (Global Maritime Overview)
- **Bản đồ số WebGL**: Hiển thị 2,424 cảng biển và 169,888 tuyến vận tải biển toàn cầu trên nền Carto Positron vector tiles thanh lịch.
- **6 Thẻ HUD trực quan**: Tổng số cảng biển (2,424), Tuyến kết nối (169,888), Mật độ mạng (4.03%), Tỷ lệ thành phần liên thông lớn nhất LCC (99.9%), Modularity Q Leiden (0.380), Cảng biển Việt Nam (85 cảng).
- **Bộ lọc đa tầng**: Tùy chọn hiển thị các lớp `ports`, `ports_glow` (Top 30 siêu cảng), `vn_stars` (sao vàng cảng VN), `arcs` (đại vòng cung), `searoutes` (tuyến biển bathymetric thực tế), `chokepoints` (7 điểm nghẽn chiến lược), và `seamark` (hải tiêu OpenSeaMap).
- **Tùy biến trực quan**: Thay đổi kích thước cảng theo 5 độ đo (PageRank, Betweenness, Degree, Closeness, k-Core) và màu sắc theo phân cụm cộng đồng hoặc phân vùng Việt Nam.
- **Hoạt ảnh Luồng tàu chạy (TripsLayer)**: Tab hoạt ảnh WebGL thời gian thực của 400 hành trình tàu với các nút bấm Play/Pause, tua tốc độ và đồng hồ hải trình.
- **Bảng xếp hạng Top 20**: Liệt kê 20 cảng lớn nhất thế giới, làm nổi bật cảng biển Việt Nam và hỗ trợ click điều hướng trực tiếp sang Trang 01.

### 🏢 Trang 01: Hồ sơ Cảng biển (Port Dossier & Ego-Network)
- **Tra cứu cảng thông minh**: Tự động nhận diện từ URL `?port=port434` hoặc ô tìm kiếm với toạ độ ECDIS độ phút giây hải đồ và LOCODE chuẩn quốc tế.
- **Biểu đồ Radar 6 độ đo**: Trực quan hoá hình thế kết nối percentile của cảng đối sánh với Top 10 trung bình cảng biển Việt Nam và siêu cảng Singapore (`port1201`).
- **Mạng lân cận (Ego-Network)**: Bản đồ Pydeck tương tác 1–2 hop với phân biệt màu sắc luồng hàng hải vào (xanh ngọc Teal) và luồng ra (xanh đại dương Oceanic).
- **Phân rã sản lượng tàu ghé**: Biểu đồ cột chồng phân loại 5 nhóm tàu chuyên dụng (Container, Hàng rời, Tàu dầu, Bách hoá, Ro-Ro) theo chuỗi thời gian của PortWatch.
- **Kiến thức SNA**: Giải thích lý thuyết Lỗ hổng Cấu trúc (Burt's Structural Holes) và chỉ số Network Constraint ($C_i$).

### 🧭 Trang 02: Tìm Tuyến Đường biển (Maritime Route Planner)
- **Thuật toán Yen's K-Shortest Loopless Paths**: Tìm từ 1 đến 5 phương án hải trình tối ưu theo tiêu chí "Nhanh nhất (số ngày)" hoặc "Ít chuyển tải nhất".
- **Bộ lọc tránh điểm nghẽn**: Tính năng phòng thủ tránh Eo biển Malacca, Kênh đào Suez/Biển Đỏ, hoặc các cảng/quốc gia phong toả tùy chọn.
- **Hải trình Bathymetric thực tế**: Vẽ tuyến đường biển uốn lượn theo luồng hàng hải thực tế từ `edges_geo.parquet` mà không cắt ngang đất liền.
- **Voyage Timeline**: Biểu đồ thanh ngang phân tách khoảng cách (km, hải lý) và thời gian (ngày) từng chặng.
- **So chuẩn thuật toán**: Khối thực nghiệm đối chiếu trực tiếp giữa thuật toán Dijkstra và thuật toán A* với hàm Heuristic Haversine admissible.

### 🧩 Trang 03: Phát hiện Cộng đồng (Community Detection)
- **6 Thuật toán phân cụm**: Louvain, Leiden, Label Propagation (LPA), Infomap, Greedy Modularity, và Girvan-Newman (phân mạng Đông Nam Á).
- **Thanh trượt Resolution $\gamma$ động**: Tái tính toán phân vùng cộng đồng Leiden/Louvain tức thì trong ~40ms khi người dùng thay đổi hệ số phân giải $\gamma \in [0.2, 3.0]$.
- **Ma trận đối sánh NMI & ARI**: Biểu đồ nhiệt Plotly Heatmap so sánh độ tương đồng phân hoạch giữa tất cả các cặp thuật toán.
- **Đồ thị lực Girvan-Newman SEA**: Trực quan hoá mạng lưới Đông Nam Á và thanh trượt từng bước cắt cạnh có Betweenness cao nhất.

### ⚠️ Trang 04: Mô phỏng Sự cố & Gián đoạn (Disruption Simulation & Robustness)
- **Kịch bản sự cố cài sẵn**: Đóng eo biển Malacca, Kênh đào Suez, đóng đồng thời Malacca + Suez, Bão nhiệt đới đổ bộ miền Bắc VN, Cảng nước sâu Cái Mép ngừng hoạt động, Tấn công có chủ đích vào Top 10 Hubs, hoặc gỡ bỏ nút tùy chọn.
- **Bản đồ kép song song**: So sánh trực quan mạng lưới trước và sau gián đoạn, làm nổi bật các cảng bị cô lập màu đỏ phát sáng.
- **Đường cong suy thoái Percolation**: Biểu đồ tương tác kích thước thành phần liên thông lớn nhất (LCC Fraction $S$) và hiệu suất toàn cầu ($E$).
- **Phân tích lan truyền Spillover**: Bản đồ nhiệt và bảng xếp hạng mức độ gián đoạn lan truyền tới hệ thống cảng biển Việt Nam theo mô hình thực nghiệm PortWatch.

### 🔮 Trang 05: Dự đoán Tuyến mới (Link Prediction)
- **Đánh giá 6 mô hình Link Prediction**: Common Neighbors, Jaccard, Adamic-Adar, Resource Allocation, Preferential Attachment, và Random Forest Classifier.
- **Đường cong ROC & PR**: Biểu đồ Plotly phân tích năng lực dự báo với diện tích dưới đường cong AUC-ROC lên tới 0.9520.
- **Gợi ý Top 10 tuyến chiến lược**: Bộ chọn cảng Việt Nam hiển thị 10 tuyến kết nối quốc tế tiềm năng nhất kèm danh sách các siêu hub trung chuyển chung.
- **Bản đồ Pydeck Amber Arcs**: Trực quan hoá các cung liên kết tiềm năng bằng đường cung đại vòng màu hổ phách phát sáng.

### 🌐 Trang 06: Việt Nam Toàn cầu & Chuỗi thời gian LSBCI (Global Liner Connectivity)
- **Chuỗi thời gian 21 quý (2006-2026)**: Thanh trượt quý tích hợp nút Play/Pause tự động chạy chuỗi thời gian chuyển dịch kết nối hàng hải.
- **Serrano Disparity Filter Backbone**: Bản đồ Pydeck lọc bỏ nhiễu ngẫu nhiên, chỉ giữ lại các liên kết thương mại xương sống có ý nghĩa thống kê ($\alpha \le 0.15$).
- **Biểu đồ Sankey Alluvial**: Minh hoạ bước chuyển dịch khối liên minh của Việt Nam từ cảng vệ tinh phụ thuộc (2006) thành siêu cụm hàng hải trung tâm ASEAN (2024).
- **Tương quan kinh tế thực**: Phân tích mối tương quan tuyến tính giữa chỉ số LSBCI và số lượt tàu ghé thực tế ($r = +0.658, p < 0.001$).

### 🚢 Trang 07: Mạng Phà Nội địa (Domestic Ferry Network)
- **Mạng lưới 25 bến phà biển & 17 tuyến đôi**: Phủ kín 4 vùng duyên hải trọng điểm: Vịnh Bắc Bộ (Hải Phòng - Cát Bà - Cô Tô), Duyên hải Miền Trung (Lý Sơn, Cù Lao Chàm, Phú Quý), Đông Nam Bộ (TP.HCM - Cần Giờ - Vũng Tàu - Côn Đảo), và Kiên Giang (Phú Quốc, Nam Du, Hòn Sơn).
- **Trình tìm đường đi phà (Point-to-point Route Planner)**: Tìm đường đi ngắn nhất giữa 2 bến phà bất kỳ, tính toán thời gian, cự ly và cảnh báo phân mảnh địa lý đối với các cụm đảo biệt lập.
- **Tự động điều chỉnh tiêu cự**: Camera tự động zoom vào vùng biển được chọn.

### 🔬 Trang 08: Cấu trúc Mạng Toàn cầu (Macro Topology & Complex Networks)
- **Kiểm định Thế giới nhỏ (Small-World)**: Chứng minh tính chất thế giới nhỏ với hệ số $\sigma = 1.500 > 1.0$ ($C = 0.0914$, $L = 2.628$ bước).
- **Kiểm định Không quy mô (Scale-Free)**: Biểu đồ Log-Log phân phối bậc và kiểm định Vuong Log-Likelihood Ratio xác nhận phân phối hàm mũ lũy thừa ($\alpha = 2.99, x_{\min} = 165$).
- **Hiện tượng Câu lạc bộ Đại gia (Rich-Club)**: Biểu đồ $\rho_{\text{norm}}(k) > 1.0$ phản ánh cấu trúc siêu liên kết giữa các cảng đầu mối và tính không đồng loại ($r = -0.051$).
- **Lỗ hổng Cấu trúc 4 góc phần tư**: Đồ thị phân tán Transit Betweenness $\times$ Burt's Constraint xác định các cảng giữ vị thế môi giới thương mại quốc tế (vinh danh Phú Mỹ, Hải Phòng, TP.HCM).
- **Phân rã vỏ lõi $k$-Core**: Xác định đúng 5 cảng biển Việt Nam lọt vào lõi cực đại thế giới ($k=120$).

### 📖 Trang 09: Phương pháp luận & Dữ liệu (Methodology, Data Audit & Lineage)
- **Sơ đồ Dòng dữ liệu Mermaid 5 tầng**: Trực quan hoá toàn vẹn đường dẫn dữ liệu từ 1.157M bản ghi thô đến 16 artifacts và giao diện tương tác.
- **Kiểm toán dữ liệu mẫu**: Bảng đối chiếu nguồn gốc từ IMF PortWatch, Global Fishing Watch, UNCTAD và OpenStreetMap.
- **Trình duyệt Manifest**: Tra cứu kích thước file, thời điểm tạo, số bản ghi và xem trước cấu trúc bảng của từng artifact.
- **Trích dẫn Học thuật Quốc tế**: Danh mục 8 công trình kinh điển của Newman, Barabási, Serrano, Burt, Watts-Strogatz, Traag, Rosvall, UNCTAD.

---

## 5. Kiểm thử Tự động & Đảm bảo Chất lượng (Testing & QA)

Dự án áp dụng quy trình kiểm thử tự động nghiêm ngặt 4 tầng (Tiers 1 đến 4) với **99 bài kiểm thử đạt 100% tỷ lệ vượt qua**:

```powershell
pytest tests/ -v
```

Kết quả thực thi kiểm thử:
- `tests/test_data_contracts.py`: **30/30 PASSED** (Kiểm tra hợp đồng dữ liệu, schema, không null, kiểu dữ liệu 16 artifacts).
- `tests/test_graph.py`: **20/20 PASSED** (Kiểm tra đồ thị vô hướng, có hướng, Dijkstra, A*, Yen's k-shortest, Haversine).
- `tests/test_algorithms.py`: **28/28 PASSED** (Kiểm tra bất biến toán học PageRank bảo toàn xác suất, Modularity Q, LCC, Vuong LR).
- `tests/test_e2e_app.py`: **21/21 PASSED** (Kiểm thử tương thích giao diện, kiểm thử không ngoại lệ Streamlit Headless AppTest xuyên suốt 10 trang).

---

## 6. Trích dẫn & Bản quyền (Citations & Licenses)

Dữ liệu và thuật toán trong dự án được xây dựng dựa trên các công trình khoa học đã được bình duyệt:
1. **UNCTAD (2006–2024)**: *Liner Shipping Bilateral Connectivity Index (LSBCI)*. United Nations Conference on Trade and Development.
2. **IMF PortWatch (2019–2026)**: *Real-time Global Vessel Port Calls & Disruptions Dataset*. International Monetary Fund.
3. **Global Fishing Watch (2024)**: *Global Marine Vessel Tracking Data*. Creative Commons Attribution-NonCommercial 4.0.
4. **Serrano, M. Á., Boguñá, M., & Vespignani, A. (2009)**: *Extracting the multiscale backbone of complex networks*. PNAS, 106(16), 6483-6488.
5. **Burt, R. S. (1992)**: *Structural Holes: The Social Structure of Competition*. Harvard University Press.
6. **Traag, V. A., Waltman, L., & van Eck, N. J. (2019)**: *From Louvain to Leiden: guaranteeing well-connected communities*. Scientific Reports, 9, 5233.
7. **Barabási, A.-L., & Albert, R. (1999)**: *Emergence of scaling in random networks*. Science, 286(5439), 509-512.
8. **Watts, D. J., & Strogatz, S. H. (1998)**: *Collective dynamics of 'small-world' networks*. Nature, 393(6684), 440-442.

---
*Đồ án Phân tích Mạng Xã hội (SNA) — Hoàn thành tháng 11/2026.*
