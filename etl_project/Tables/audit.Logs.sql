CREATE OR ALTER TABLE audit.Logs (
  IdLog 			int IDENTITY(1,1) PRIMARY KEY,
  IdLogPadre 		int,
  IdLogCarga		int,
  FechaInicio		datetime,
  FechaFin			datetime,
  Procedimiento 		varchar(100),
  NumRegistros		int,
  DuracionSegundos	int,
  MensajeError		varchar(max),
  Usuario			varchar(50),
  Nivel				int,
  EsPadre			bit,
  Spid				varchar(5),
  Hash				varchar(100)
)


CREATE OR REPLACE EXTENDED PROPERTY AuditSchema ON DATABASE AS 'audit'
CREATE OR REPLACE EXTENDED PROPERTY AuditTable ON DATABASE AS 'Logs'


CREATE OR REPLACE EXTENDED PROPERTY AuditIdColumn 				ON COLUMN audit.Logs(IdLog) 
CREATE OR REPLACE EXTENDED PROPERTY AuditParentIdColumn 		ON COLUMN audit.Logs(IdLogPadre) 
CREATE OR REPLACE EXTENDED PROPERTY AuditRootIdColumn 			ON COLUMN audit.Logs(IdLogCarga) 
CREATE OR REPLACE EXTENDED PROPERTY AuditStartDateColumn 		ON COLUMN audit.Logs(FechaInicio) 
CREATE OR REPLACE EXTENDED PROPERTY AuditEndDateColumn 			ON COLUMN audit.Logs(FechaFin) 
CREATE OR REPLACE EXTENDED PROPERTY AuditDescriptionColumn 		ON COLUMN audit.Logs(Procedimiento) 
CREATE OR REPLACE EXTENDED PROPERTY AuditRowsCountColumn 		ON COLUMN audit.Logs(NumRegistros)
CREATE OR REPLACE EXTENDED PROPERTY AuditMessageErrorColumn 	ON COLUMN audit.Logs(MensajeError)
CREATE OR REPLACE EXTENDED PROPERTY AuditUserColumn 			ON COLUMN audit.Logs(Usuario) 
CREATE OR REPLACE EXTENDED PROPERTY AuditSpidColumn 			ON COLUMN audit.Logs(Spid)
