WITH tb_life_cycle_atual AS (

    SELECT IdCliente,
           qtdeFrequencia,
           descLifeCycle AS descLifeCycleAtual

    FROM life_cycle
    WHERE dtRef = date('{date}','-1 day')

),

tb_life_cycle_D28 AS (

    SELECT IdCliente,
        descLifeCycle AS descLifeCycleD28

    FROM life_cycle
    WHERE dtRef = date('{date}','-29 day')
),

tb_share_ciclos AS (

    SELECT idCliente,
            1. * SUM(CASE WHEN descLifeCycle = '01-CURIOSO' THEN 1 ELSE 0 END) / COUNT(*) AS pctCurioso,
            1. * SUM(CASE WHEN descLifeCycle = '02-FIEL' THEN 1 ELSE 0 END) / COUNT(*) AS pctFiel,
            1. * SUM(CASE WHEN descLifeCycle = '03-TURISTA' THEN 1 ELSE 0 END) / COUNT(*) AS pctTurista,
            1. * SUM(CASE WHEN descLifeCycle = '04-DESENCANTADA' THEN 1 ELSE 0 END) / COUNT(*) AS pctDesencantada,
            1. * SUM(CASE WHEN descLifeCycle = '05-ZUMBI' THEN 1 ELSE 0 END) / COUNT(*) AS pctZumbi,
            1. * SUM(CASE WHEN descLifeCycle = '02-RECONQUISTADO' THEN 1 ELSE 0 END) / COUNT(*) AS pctReconquistado,
            1. * SUM(CASE WHEN descLifeCycle = '02-REBORN' THEN 1 ELSE 0 END) / COUNT(*) AS pctReborn

    FROM life_cycle
    WHERE dtRef < '{date}'

    GROUP BY idCliente

),


tb_avg_ciclo AS (

    SELECT descLifeCycleAtual,
          AVG(qtdeFrequencia) AS avgFreqGrupo

    FROM tb_life_cycle_atual

    GROUP BY descLifeCycleAtual

),

tb_join AS (

    SELECT t1.*,
        t2.descLifeCycleD28,
        t3.pctCurioso,
        t3.pctFiel,
        t3.pctTurista,
        t3.pctDesencantada,
        t3.pctZumbi,
        t3.pctReconquistado,
        t3.pctReborn,
        t4.avgFreqGrupo,
        1.* t1.qtdeFrequencia / t4.avgFreqGrupo AS ratioFreqGrupo

    FROM tb_life_cycle_atual AS t1
    LEFT JOIN tb_life_cycle_D28 AS t2
    ON t1.IdCliente = t2.IdCliente

    LEFT JOIN tb_share_ciclos AS t3
    ON t1.idCliente = t3.idCliente

    LEFT JOIN tb_avg_ciclo AS t4
    ON t1.descLifeCycleAtual = t4.descLifeCycleAtual

)


SELECT date('{date}', '-1 day') AS dtRef,
       *

FROM tb_join