// 2019/09/01 Hiroyuki Ogasawara
// vim:ts=4 sw=4 noet:

#ifndef	CCTFLIBRARY_H_
#define	CCTFLIBRARY_H_

#include	<assert.h>
#include	<tensorflow/c/c_api.h>
#include	<cctf/CCTFShape.h>


namespace cctf {
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

#if 0
template<typename T> constexpr TF_DataType	CTypeToTFType	= TF_FLOAT;
template<> constexpr TF_DataType	CTypeToTFType<float>	= TF_FLOAT;
template<> constexpr TF_DataType	CTypeToTFType<double>	= TF_DOUBLE;
template<> constexpr TF_DataType	CTypeToTFType<int>		= TF_INT32;
template<> constexpr TF_DataType	CTypeToTFType<uint8_t>	= TF_UINT8;
template<> constexpr TF_DataType	CTypeToTFType<int16_t>	= TF_INT16;
template<> constexpr TF_DataType	CTypeToTFType<int8_t>	= TF_INT8;
template<> constexpr TF_DataType	CTypeToTFType<uint16_t>	= TF_UINT16;
#else
template<typename T>
struct CTypeToTFType {};
template<> struct CTypeToTFType<float>		{ static constexpr TF_DataType val= TF_FLOAT; };
template<> struct CTypeToTFType<double>		{ static constexpr TF_DataType val= TF_DOUBLE; };
template<> struct CTypeToTFType<int>		{ static constexpr TF_DataType val= TF_INT32; };
template<> struct CTypeToTFType<uint8_t>	{ static constexpr TF_DataType val= TF_UINT8; };
template<> struct CTypeToTFType<int16_t>	{ static constexpr TF_DataType val= TF_INT16; };
template<> struct CTypeToTFType<int8_t>		{ static constexpr TF_DataType val= TF_INT8; };
template<> struct CTypeToTFType<uint16_t>	{ static constexpr TF_DataType val= TF_UINT16; };
#endif


//-----------------------------------------------------------------------------

template<typename T>
class CCPointer {
protected:
	T*	iPointer= nullptr;
	CCPointer& operator=( const CCPointer& )= delete;
	CCPointer( const CCPointer& )= default;
public:
	CCPointer()
	{
	}
	CCPointer( CCPointer&& src )
	{
		iPointer= src;
		src.iPointer= nullptr;
	}
	~CCPointer()
	{
	}
	T*	IPointer() const
	{
		assert( iPointer != nullptr );
		return	iPointer;
	}
	T*	nIPointer() const
	{
		return	iPointer;
	}
	void	SetPointer( T* ptr )
	{
		assert( iPointer == nullptr );
		iPointer= ptr;
	}
};


//-----------------------------------------------------------------------------

class CCStatus : public CCPointer<TF_Status> {
	CCStatus( const CCStatus& )= delete;
	CCStatus& operator=( const CCStatus& )= delete;
public:
	CCStatus();
	~CCStatus();
	void	Finalize();
	bool	IsOK() const;
	bool	IsError() const;
};


//-----------------------------------------------------------------------------

class CCBuffer : public CCPointer<TF_Buffer> {
	CCBuffer( const CCBuffer& )= delete;
	CCBuffer& operator=( const CCBuffer& )= delete;
protected:
	static void	cb_Deallocator( void* data, size_t size );
public:
	CCBuffer();
	~CCBuffer();
	void	Finalize();
	const void*	GetData() const;
	size_t	GetDataSize() const;
	template<typename T>
	const T*	Map() const
	{
		return	reinterpret_cast<const T*>(GetData());
	}
	bool	LoadFile( const char* path );
	bool	SaveFile( const char* path );
};


//-----------------------------------------------------------------------------

class CCTensor : public CCPointer<TF_Tensor> {
	CCTensor( const CCTensor& )= delete;
	CCTensor& operator=( const CCTensor& )= delete;
protected:
	static void	cb_Deallocator( void* data, size_t size, void* arg );
public:
	CCTensor();
	CCTensor( TF_DataType type, const CCShape& shape );
	~CCTensor();
	void	Finalize();
	void	Allocate( TF_DataType type, const CCShape& shape );
	void	SetData( TF_DataType type, const CCShape& shape, const void* data, size_t data_size );
	template<typename T>
	void	Set( const CCShape& shape, const T* data, size_t count )
	{
		SetData( CTypeToTFType<T>::val, shape, data, sizeof(T) * count );
	}
	bool	SetString( const char* text );
	TF_DataType	GetType() const;
	size_t	GetByteSize() const;
	int64_t	GetElementCount() const;
	void*	GetData() const;
	template<typename T>
	T*	Map() const
	{
		return	reinterpret_cast<T*>(GetData());
	}
	void	GetShape( CCShape& shape ) const;
	void	Dump( const char* text= "" ) const;
};


//-----------------------------------------------------------------------------

class CCOperation : public CCPointer<TF_Operation> {
public:
	CCOperation();
	~CCOperation();
	void	Finalize();
	int		GetInputCount() const;
	int		GetOutputCount() const;
	const char*	GetDevice() const;
	void	Dump_Input();
	void	Dump_Output();
	void	Dump();
};


//-----------------------------------------------------------------------------

struct CCOutput {
	TF_Output	Output;
public:
	CCOutput( const CCOperation& op, int index ) : Output{ op.IPointer(), index }
	{
	}
};


//-----------------------------------------------------------------------------

class CCOperationDescription : public CCPointer<TF_OperationDescription> {
public:
	CCOperation*	iOperation= nullptr;
public:
	CCOperationDescription( TF_OperationDescription*, CCOperation* );
	~CCOperationDescription();
	CCOperationDescription&	AddInput( const CCOperation& src );
	CCOperationDescription&	SetAttr( const char* attr_name, CCTensor& tensor );
	CCOperationDescription&	SetAttr( const char* attr_name, TF_DataType type );
	CCOperationDescription&	SetDevice( const char* device );
	bool	Finish();
};


//-----------------------------------------------------------------------------

class CCImportGraphDefOptions : public CCPointer<TF_ImportGraphDefOptions> {
	CCImportGraphDefOptions( const CCImportGraphDefOptions& )= delete;
	CCImportGraphDefOptions& operator=( const CCImportGraphDefOptions& )= delete;
public:
	CCImportGraphDefOptions();
	~CCImportGraphDefOptions();
	void	Finalize();
	CCImportGraphDefOptions&	SetDefaultDevice( const char* device );
};


//-----------------------------------------------------------------------------

/*!
	@code
		CCGraph	graph;
		CCOperation	op;
		graph.CreateOperation( op, "Add", "None" )
			.AddInput( src1 )
			.AddInput( src2 )
			.Finish();

		graph.CreateOperation( op, "Const", "a" )
			.SetAttr( "value", tensor )
			.SetAttr( "dtype", tensor.GetType() )
			.Finish();
	@endcode

	@code
		CCBuffer	buffer;
		buffer.LoadFile( "aaaaa.pb" );
		CCGraph	graph;
		graph.Import( buffer, CCImportGraphDefOptions() );
	@endcode
*/
class CCGraph : public CCPointer<TF_Graph> {
	CCGraph( const CCGraph& )= delete;
	CCGraph& operator=( const CCGraph& )= delete;
public:
	CCGraph();
	~CCGraph();
	void	Finalize();
	CCOperationDescription	CreateOperation( CCOperation& op, const char* op_type, const char* name );
	bool	Import( const CCBuffer& buffer, const CCImportGraphDefOptions& opt );
	bool	Export( CCBuffer& buffer );
	bool	FindOperation( CCOperation& op, const char* name );
	void	Dump();
};


//-----------------------------------------------------------------------------

class CCSessionOptions : public CCPointer<TF_SessionOptions> {
	CCSessionOptions( const CCSessionOptions& )= delete;
	CCGraph& operator=( const CCSessionOptions& )= delete;
public:
	CCSessionOptions();
	~CCSessionOptions();
	void	Finalize();
};


//-----------------------------------------------------------------------------

struct CCRunParam {
	const CCOperation*	iOp= nullptr;
	CCTensor*	iTensor= nullptr;
	int		Index= 0;
public:
	CCRunParam( const CCOperation& op, int index, CCTensor& tensor ) :
		iOp( &op ), iTensor( &tensor ), Index( index )
	{
	}
};

/*!
	@code
		CCGraph		graph;
		CCSession	session;
		session.Init( graph, CCSessionOptions() );
	@endcode
*/
class CCSession : public CCPointer<TF_Session> {
	CCSession( const CCSession& )= delete;
	CCGraph& operator=( const CCSession& )= delete;
public:
	CCSession();
	~CCSession();
	void	Finalize();
	bool	Init( CCGraph& graph, const CCSessionOptions& options );
	bool	Run( const CCOperation*const* op_list, int op_count,
					const CCRunParam* in_list, int in_count,
					const CCRunParam* out_list, int out_count );
};


//-----------------------------------------------------------------------------






//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
}

#endif

