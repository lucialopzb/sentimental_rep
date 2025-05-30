

CREATE OR ALTER TABLE audit.LogStatements (
  IdStatement		int IDENTITY(1,1) PRIMARY KEY,
  IdLog 			int NONUNIQUE	REFERENCES audit.Logs,
  FechaInicio 		datetime,
  FechaFin			datetime,
  Procedimiento		varchar(100),
  NumeroSentencia 	int,
  NumeroRegistros	int,
  DuracionSegundos	int,
  MensajeError		varchar(max),
  CronoHash			varchar(100),
  Hash				varchar(100) 
)
