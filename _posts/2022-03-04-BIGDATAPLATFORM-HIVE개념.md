---
layout: post
title:  "HADOOP, HIVE, HUE"
date:   2021-03-18T14:25:52-05:00
author: KSJ
categories: BIGDATAPLATFORM
---
## 구조
로컬 -> R 작업 디렉토리 -> HADOOP

## HADOOP 구성요소
데이터노드:
네임노드:
엣지노드:

## HADOOP & HIVEQUERY


## HIVE
DROP TABLE TEST1
HIVE는 Hadoop 에코시스템에서 데이터 핸들링/모델링하는 경우, 가장 많이 사용하는 데이터 웨어하징용 솔루션
RDB의 데이터베이스, 테이블과 같은 형태로 HDFS에 저장된 데이터의 구조를 정의하는 방법을 제공하며,
SQL과 유사한 HiveQL 쿼리를 이용하여 데이터를 조회할 수 있습니다.


# RDB(Relational Database)
 RDB는 관계형 데이터 모델에 기초를 둔 Database입니다. 
 관계형 데이터 모델은 모든 데이터를 2차원의 테이블 형태로 표현합니다.
 관계형 데이터베이스는 키와 값들의 간단한 관계를 테이블화 시킨 매우 간단한 원칙의 전산정보 데이터베이스라고 합니다.

# HDFS(Hadoop File System)
HDFS는 하둡에서 실행되는 파일을 관리해주는 시스템으로, 크게 NameNode, DataNode로 구성되어 있습니다.

- NameNode는 HDFS에서 핵심적이고 중요한 역할을 하는데요.

파일 속성 정보, 파일 시스템 정보 등의 메타데이터 관리를 합니다.
디스크가 아닌 메모리에서 직접 관리하여 속도가 빠릅니다.

## JDBC(Java Database Connectivity)
JDBC는 자바에서 DB에 접속할 수 있도록 하는 자바 API라고 합니다
JDBC는 DB에서 자료를 쿼리하거나 업데이트하는 방법을 제공합니다.

## ODBC(OPEN DATABASE CONNECTIVITY)
ODBC는 표준 개방형 응용 프로그램 인터페이스입니다.
MS에서 만든 DB에 접근하기 위한 소프트웨어 표준 규격입니다.
ODBC를 사용하면 여러 종류의 DB에 접근할 수 있습니다.

## R
- dbGetquery
- TABLE 생성시 FORMAT지정 필수


# HUE
Hadoop 에코시스템을 통합하여 보여주는 솔루션 
