SELECT dtRef,
       descLifeCycle,
       count(*) AS qtdeCliente

FROM life_cycle

WHERE descLifeCycle <> '05-ZUMBI'
AND dtRef = (SELECT MAX(dtRef) FROM life_cycle)

group by dtRef, descLifeCycle
order by dtRef, descLifeCycle