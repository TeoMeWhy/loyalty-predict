WITH tb_usuario_cursos AS (
    SELECT idUsuario,
           descSlugCurso,
           COUNT(descSlugCursoEpisodio) AS qtdeEps

    FROM cursos_episodios_completos
    WHERE DtCriacao < '{date}'
    GROUP BY idUsuario, descSlugCurso
),

tb_cursos_total_eps AS (

    SELECT descSlugCurso,
           COUNT(descEpisodio) as qtdeTotalEps
    FROM cursos_episodios
    GROUP BY descSlugCurso

),

tb_pct_cursos AS (

    SELECT t1.idUsuario,
            t1.descSlugCurso,
            1. * t1.qtdeEps / t2.qtdeTotalEps AS pctCursoCompleto

    FROM tb_usuario_cursos AS t1

    LEFT JOIN tb_cursos_total_eps AS t2
    ON t1.descSlugCurso = t2.descSlugCurso

),

tb_pct_cursos_pivot AS (

    SELECT idUsuario,

        SUM(CASE WHEN pctCursoCompleto = 1 then 1 else 0 end) AS qtdeCursosCompletos,
        SUM(CASE WHEN pctCursoCompleto > 0 and pctCursoCompleto < 1 then 1 else 0 end) AS qtdeCursosIncompletos,
        SUM(CASE WHEN descSlugCurso = 'carreira' THEN pctCursoCompleto ELSE 0 END) AS carreira,
        SUM(CASE WHEN descSlugCurso = 'coleta-dados-2024' THEN pctCursoCompleto ELSE 0 END) AS coletaDados2024,
        SUM(CASE WHEN descSlugCurso = 'ds-databricks-2024' THEN pctCursoCompleto ELSE 0 END) AS dsDatabricks2024,
        SUM(CASE WHEN descSlugCurso = 'ds-pontos-2024' THEN pctCursoCompleto ELSE 0 END) AS dsPontos2024,
        SUM(CASE WHEN descSlugCurso = 'estatistica-2024' THEN pctCursoCompleto ELSE 0 END) AS estatistica2024,
        SUM(CASE WHEN descSlugCurso = 'estatistica-2025' THEN pctCursoCompleto ELSE 0 END) AS estatistica2025,
        SUM(CASE WHEN descSlugCurso = 'github-2024' THEN pctCursoCompleto ELSE 0 END) AS github2024,
        SUM(CASE WHEN descSlugCurso = 'github-2025' THEN pctCursoCompleto ELSE 0 END) AS github2025,
        SUM(CASE WHEN descSlugCurso = 'ia-canal-2025' THEN pctCursoCompleto ELSE 0 END) AS iaCanal2025,
        SUM(CASE WHEN descSlugCurso = 'lago-mago-2024' THEN pctCursoCompleto ELSE 0 END) AS lagoMago2024,
        SUM(CASE WHEN descSlugCurso = 'machine-learning-2025' THEN pctCursoCompleto ELSE 0 END) AS machineLearning2025,
        SUM(CASE WHEN descSlugCurso = 'matchmaking-trampar-de-casa-2024' THEN pctCursoCompleto ELSE 0 END) AS matchmakingTramparDeCasa2024,
        SUM(CASE WHEN descSlugCurso = 'ml-2024' THEN pctCursoCompleto ELSE 0 END) AS ml2024,
        SUM(CASE WHEN descSlugCurso = 'mlflow-2025' THEN pctCursoCompleto ELSE 0 END) AS mlflow2025,
        SUM(CASE WHEN descSlugCurso = 'pandas-2024' THEN pctCursoCompleto ELSE 0 END) AS pandas2024,
        SUM(CASE WHEN descSlugCurso = 'pandas-2025' THEN pctCursoCompleto ELSE 0 END) AS pandas2025,
        SUM(CASE WHEN descSlugCurso = 'python-2024' THEN pctCursoCompleto ELSE 0 END) AS python2024,
        SUM(CASE WHEN descSlugCurso = 'python-2025' THEN pctCursoCompleto ELSE 0 END) AS python2025,
        SUM(CASE WHEN descSlugCurso = 'sql-2020' THEN pctCursoCompleto ELSE 0 END) AS sql2020,
        SUM(CASE WHEN descSlugCurso = 'sql-2025' THEN pctCursoCompleto ELSE 0 END) AS sql2025,
        SUM(CASE WHEN descSlugCurso = 'streamlit-2025' THEN pctCursoCompleto ELSE 0 END) AS streamlit2025,
        SUM(CASE WHEN descSlugCurso = 'trampar-lakehouse-2024' THEN pctCursoCompleto ELSE 0 END) AS tramparLakehouse2024,
        SUM(CASE WHEN descSlugCurso = 'tse-analytics-2024' THEN pctCursoCompleto ELSE 0 END) AS tseAnalytics2024

    FROM tb_pct_cursos

    GROUP BY idUsuario  

),

tb_atividade AS (

    SELECT 
        idUsuario,
        max(dtRecompensa) as dtCriacao

    FROM recompensas_usuarios
    WHERE dtRecompensa < '{date}'
    GROUP BY idUsuario

    UNION ALL

    SELECT 
        idUsuario,
        max(dtCriacao) AS dtCriacao

    FROM habilidades_usuarios
    WHERE DtCriacao < '{date}'
    GROUP BY idUsuario

    UNION ALL

    SELECT
        idUsuario,
        max(dtCriacao) AS dtCriacao

    FROM cursos_episodios_completos
    WHERE DtCriacao < '{date}'
    GROUP BY idUsuario

),

tb_ultima_atividade AS (

    SELECT idUsuario,
            MIN( julianday('{date}') - julianday(dtCriacao)) AS qtdDiasUltiAtividade

    FROM tb_atividade

    GROUP BY idUsuario

),

tb_join AS (

    SELECT t3.idTMWCliente AS idCliente,
        t1.qtdeCursosCompletos,
        t1.qtdeCursosIncompletos,
        t1.carreira,
        t1.coletaDados2024,
        t1.dsDatabricks2024,
        t1.dsPontos2024,
        t1.estatistica2024,
        t1.estatistica2025,
        t1.github2024,
        t1.github2025,
        t1.iaCanal2025,
        t1.lagoMago2024,
        t1.machineLearning2025,
        t1.matchmakingTramparDeCasa2024,
        t1.ml2024,
        t1.mlflow2025,
        t1.pandas2024,
        t1.pandas2025,
        t1.python2024,
        t1.python2025,
        t1.sql2020,
        t1.sql2025,
        t1.streamlit2025,
        t1.tramparLakehouse2024,
        t1.tseAnalytics2024,
        t2.qtdDiasUltiAtividade

    FROM tb_pct_cursos_pivot as t1

    LEFT JOIN tb_ultima_atividade AS t2
    ON t1.idUsuario = t2.idUsuario

    INNER JOIN usuarios_tmw AS t3
    ON t1.idUsuario = t3.idUsuario

)

SELECT date('{date}', '-1 day') AS dtRef,
        *
FROM tb_join