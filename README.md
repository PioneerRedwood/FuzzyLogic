개요:        인공지능 과제 #4

제목:        Fuzzy Theory, 퍼지 이론

기간:        2020.6.29 ~ 2020.7.2

소유자:      ChrisRedwood


주요 용어

    퍼지 집합, 소속 함수
    
    퍼지화
    
    퍼지추론 규칙 기반
    
    역퍼지화 무게 중심법


설명

    본 프로그램은 대한민국 2019년 7월 1일부터 2019년 8월 31일 집계된 온도와 습도 데이터로
    근로자들의 적절한 휴식 시간을 찾는 것이 목적이다.


수정 로그

    #1


    #2
    - Fuzzy.py                           메인 부분 클릭 이벤트 함수 수정
    - STCS_190701_190831_forNP.csv       csv 파일 numpy.loadtxt() 읽어오는 파일 업로드


    #3
    2020.07.02.
    - Environment.py 그래프 그려지는 부분 퍼지 집합에 맞게 영역 표시되도록 수정
                     온도 및 습도 퍼지 집합 수정
                     퍼지 교집합 연산 및 퍼지 합집합 연산 주석 수정
