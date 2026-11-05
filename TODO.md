# 작업 현황

최종 점검: 2026-11-05. 기능 구현 없이 상태 문서만 정리했다.
기존 `TODO.md`, `docs/TODO.md`, 레포 내부 `AGENTS.md`·`CLAUDE.md`는 없었다. 기존 작업 목록은 README의 개선 계획이며, 이를 보존해 아래에 구분했다.
기준 커밋은 `c58d27b`(일괄 보정)와 `47887f8`(리뷰·수동 조정·자동 크롭)이다. 시작 시 추적·미추적 변경은 없었다.

## 완료: 구현·산출물 확인

아래 체크는 코드와 파일의 정적 대조 결과다. 실행 검증 상태는 보류·막힘 구역에 별도로 기록한다.

- [x] 일괄 기울기 보정 CLI 구현 확인: `auto_level.py`의 `estimate_skew_angle`, `process_image`, `process_directory`, `find_images`, `build_parser`에 각도 추정·저장·실패 수집·재귀 검색이 있다.
- [x] 회전 후 자동 크롭 구현 확인: `rotate_image(crop=True)`와 `_largest_valid_rectangle`이 유효 영역을 잘라낸다. `crop=False`는 캔버스 확장을 유지하며, 원본 종횡비 유지 옵션은 아직 없다.
- [x] OpenCV 전후 비교·수동 조정 구현 확인: `make_review_canvas`, `review_directory`와 `--review`·`--angle-step`이 있다. 실제 창·키 입력 동작은 미검증이다.
- [x] 합성 이미지 회귀 테스트 5개 존재 확인: `tests/test_auto_level.py`에 각도 정규화·기울기 추정·캔버스 확장·기본 크롭·일괄 저장 사례가 있다. 통과 여부는 확인하지 못했다.
- [x] 개인 사진·결과물 제외 설정 확인: `.gitignore`에 `input/*`, `output/*`와 `.gitkeep` 예외가 있고, 추적 목록에는 개인 사진·결과물이 없다.
- [x] 상태 문서 정리: README의 자동 크롭 계획을 구현 완료로 바로잡고, 남은 원본 종횡비 옵션을 유지했다. `AGENTS.md`에 TODO 갱신·동일 가지 커밋·최종 보고 규칙을 추가했다.

## 진행 중

- 진행 중인 기능 작업은 확인되지 않았다. 이번 점검에서는 기능 코드를 변경하지 않았다.

## 남은 일: 기존 README 개선 계획

코드에서 해당 구현을 확인하지 못했다. 기존 목록 순서를 유지하며 우선순위나 일정을 새로 정하지 않는다.

- [ ] 선분의 위치·길이·명암 대비를 함께 평가하는 지평선 후보 선택 개선.
- [ ] 여러 후보 선의 군집화와 신뢰도 점수 제공.
- [ ] EXIF 방향 정보와 색상 프로파일 보존 검토.
- [ ] 회전 후 원본 종횡비 유지 옵션 추가. 자동 크롭 구현과 분리한 잔여 항목이다.
- [ ] CLAHE·적응형 임계값 등 조명 환경별 전처리 옵션 제공.
- [ ] 결과 미리보기와 수동 각도 조정용 웹 UI 추가. 현재 구현은 OpenCV 데스크톱 창이다.

## 보류: 실제 화면 검증

- [ ] OpenCV 리뷰 창의 전후 비교·각도 조정·초기화·저장·원본 저장·건너뛰기·종료를 실제로 조작해 확인. 이번에는 GUI를 실행하지 않았으며, 의존성 설치와 데스크톱 창을 조작할 수 있는 환경이 준비되면 확인한다.

## 막힘: 실행 검증

- [ ] `python -m unittest discover -s tests -v` 재실행: 현재 Python 3.10.4에서 `ModuleNotFoundError: No module named 'cv2'`로 테스트 모듈 로딩이 실패했다. 5개 테스트의 성공·실패를 판정할 수 없다.
- [ ] `python auto_level.py --help` 실행 확인: 같은 `cv2` 누락으로 시작하지 못했다.
- 의존성 준비 시도: 격리된 `.venv`에서 `pip install --disable-pip-version-check --retries 0 --timeout 15 -r requirements.txt`를 실행했으나 `numpy>=1.24,<2`에 대해 `No matching distribution found`가 발생했다. 원인은 확정하지 않았다. 요구 버전의 NumPy·OpenCV를 설치할 수 있는 환경에서 위 명령을 다시 검증해야 한다.
- 확인된 검증: `python -m py_compile auto_level.py tests/test_auto_level.py` 성공. 문법 검사 결과이며 이미지 처리·GUI 실행 검증은 남아 있다.

## 구역별 대조 근거

| 점검 구역 | 확인 근거와 처리 |
| --- | --- |
| README 소개·목적·문제 정의·기술 스택 | `requirements.txt`와 Canny·Hough·weighted median 구현을 대조했다. |
| README 동작 원리·실행 방법·수동 조정 | CLI 파서와 처리·리뷰 함수를 대조하고, 회전 설명에 기본 크롭을 반영했다. 실행 검증은 위와 같이 막힘·보류다. |
| README 폴더 구조·테스트·Git 관리 원칙 | 실제 파일 목록·5개 테스트·`.gitignore`·Git 추적 목록을 확인했다. |
| README 한계점·개선 계획 | 자동 크롭만 완료 구현으로 분리했다. 다른 개선 계획 6개는 미완료로 보존했다. |
| examples/README.md | 일괄 처리 명령과 개인 사진 제외 안내를 CLI·`.gitignore`와 대조했다. 예시 이미지·데모 산출물은 없으며 완료로 기록하지 않았다. |
| 기존 TODO·작업 지침·최근 변경 | TODO와 레포 내부 지침은 없었고, 최근 두 커밋과 깨끗한 시작 상태를 확인했다. 이 파일과 `AGENTS.md`를 만들었다. |
| 완료·진행 중·남은 일·보류·막힘 | 기존에 이 상태 구역들은 없었다. 구현 확인, 기존 계획, 미실행 GUI, 의존성 때문에 막힌 명령을 각각 구분했다. |
