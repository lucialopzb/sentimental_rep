



DECLARE DEFAULT logger crono (@logPadre int=null)
BEGIN PROCEDURE INITIALIZATION


SQL `
/* ---------------------------------------------------------------------------------------------------------------------
   Este procedimiento se ha creado automáticamente con una herramienta.
  
   Crono ETL automatiza la generación de código SQL en entornos Data Warehouse, incluyendo 
   la modelización del DWH, la optimización de cargas completas e incrementales, y la generación de logs y documentación. 
  
   Modifique este procedimiento solo con CRONO ETL.
 -----------------------------------------------------------------------------------------------------------------------*/
`

  DECLARE @statementId int;
  DECLARE @rowcount int;


  SET NOCOUNT ON
  PRINT 'Ejecutando ' + @procedure_name
  INSERT INTO audit.Logs
  SELECT 
    @logPadre 					IdLogPadre,
    @procedure_name 			procedimiento,
	getdate() 					FechaInicio,
	cast(@@spid as varchar(5)) 	spid,
	SUSER_NAME() 				usuario,
	@procedure_hash				hash;
	
  DECLARE @log int=@@identity;
  DECLARE @logCarga int =(SELECT IdLogCarga from audit.Logs where IdLog=@logPadre);
  DECLARE @nivel int =   (SELECT nivel         from audit.Logs where IdLog=@logPadre);
  UPDATE audit.Logs SET IdLogCarga=coalesce(@logCarga,@log), nivel=coalesce(@nivel+1,0) where IdLog=@log

	
  DECLARE @Continuar bit =0;
  
  
  	   
    
sql `   BEGIN TRY 	`
SQL `
/* ---------------------------------------------------------------
  A continuación comienza la parte principal del procedimiento
------------------------------------------------------------------*/
`
  
  

end

begin statement initialization

	INSERT INTO audit.Logstatements
	SELECT 
		@log 					IdLog,
		getdate() 				FechaInicio,
		@procedure_name 		Procedimiento,
		@statement_count 		NumeroSentencia,	
		@cronostatement_hash 	CronoHash,		
		@statement_hash 		Hash;
 	SET @statementId=@@identity;

end
begin statement finalization

	SET @rowcount=@@ROWCOUNT

	UPDATE audit.Logstatements
	SET FechaFin=getdate(),DuracionSegundos=datediff(second,FechaInicio,getdate()), NumeroRegistros=@rowcount
	WHERE IdStatement=@statementId


end



begin procedure finalization

SQL `
/* ---------------------------------------------------------------
  El procedimiento termina con el registro de logs y la gestión de errores.
------------------------------------------------------------------*/
`


	update audit.Logs
	SET 
      NumRegistros=@@ROWCOUNT,
      DuracionSegundos=datediff(second,FechaInicio,getdate()),
	  FechaFin=getdate()
	WHERE IdLog=@log

	update audit.Logs
	select 
		@log #IdLog NO_GROUP,
		if(count(*)=0,0,1) EsPadre,
	from audit.Logs
	where logs.IdLogPadre = @log

sql` END TRY`
 
 
 
 
 sql` BEGIN CATCH` 

	  UPDATE audit.Logs 
	  set 
	    FechaFin=getdate(),
		MensajeError=ERROR_MESSAGE()
		where IdLog=@log;		
	  
		update audit.Logs
		select 
			@log	#IdLog NO_GROUP,
			if(count(*)=0,0,1) EsPadre,
		from audit.Logs
		where logs.IdLogPadre = @log	
		
	  IF @Continuar=0 THROW;
	
sql`	END CATCH`	
	
	
end;



