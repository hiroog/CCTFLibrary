// 2019/09/01 Hiroyuki Ogasawara
// vim:ts=4 sw=4 noet:

#include	"CCTFShape.h"
#include	"CCTFSystem.h"
#include	<assert.h>


namespace cctf {
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


CCShape::CCShape()
{
}

CCShape::CCShape( int x ) : Rank( 1 )
{
	Dimension[0]= x;
}

CCShape::CCShape( int x, int y ) : Rank( 2 )
{
	Dimension[0]= x;
	Dimension[1]= y;
}

CCShape::CCShape( int x, int y, int z ) : Rank( 3 )
{
	Dimension[0]= x;
	Dimension[1]= y;
	Dimension[2]= z;
}

CCShape::CCShape( int x, int y, int z, int w ) : Rank( 4 )
{
	Dimension[0]= x;
	Dimension[1]= y;
	Dimension[2]= z;
	Dimension[3]= w;
}

CCShape::CCShape( int x, int y, int z, int w, int u ) : Rank( 5 )
{
	Dimension[0]= x;
	Dimension[1]= y;
	Dimension[2]= z;
	Dimension[3]= w;
	Dimension[4]= u;
}

CCShape::CCShape( int x, int y, int z, int w, int u, int v ) : Rank( 6 )
{
	Dimension[0]= x;
	Dimension[1]= y;
	Dimension[2]= z;
	Dimension[3]= w;
	Dimension[4]= u;
	Dimension[5]= v;
}

CCShape::~CCShape()
{
}

void	CCShape::SetDims( const int64_t* dims, unsigned int rank )
{
	assert( rank <= RANK_MAX );
	Rank= rank;
	for( unsigned int ri= 0 ; ri< rank ; ri++ ){
		Dimension[ri]= static_cast<int>( dims[ri] );
	}
}

unsigned int	CCShape::GetDims( int64_t* dims, unsigned int rank ) const
{
	assert( rank <= RANK_MAX );
	assert( rank >= Rank );
	for( unsigned int ri= 0 ; ri< Rank ; ri++ ){
		dims[ri]= Dimension[ri];
	}
	return	Rank;
}

size_t	CCShape::GetElementCount() const
{
	size_t	count= 1;
	unsigned int	rcount= Rank;
	for( unsigned int ri= 0 ; ri< rcount ; ri++ ){
		int	dim= Dimension[ri];
		if( dim != -1 ){
			count*= Dimension[ri];
		}
	}
	return	count;
}

void	CCShape::Dump( const char* text ) const
{
	if( Rank == 0 ){
		CCSystem::Print( "%s () Scalar\n" );
		return;
	}
	CCSystem::Print( "%s (" );
	for( unsigned int ri= 0 ; ri< Rank ; ri++ ){
		CCSystem::Print( "%d,", Get( ri ) );
	}
	CCSystem::Print( ") Rank=%d Element=%zd\n", Rank, GetElementCount() );
}



//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
}

