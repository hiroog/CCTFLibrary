// 2019/09/01 Hiroyuki Ogasawara
// vim:ts=4 sw=4 noet:

#ifndef	CCTFSHAPE_H_
#define	CCTFSHAPE_H_

#include	<stdint.h>
#include	<stddef.h>


namespace cctf {
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

class CCShape {
public:
	enum : unsigned int {
		RANK_MAX	=	6,
	};
public:
	int		Dimension[RANK_MAX];
	unsigned int	Rank= 0;
public:
	CCShape();
	CCShape( int x );
	CCShape( int x, int y );
	CCShape( int x, int y, int z );
	CCShape( int x, int y, int z, int w );
	CCShape( int x, int y, int z, int w, int u );
	CCShape( int x, int y, int z, int w, int u, int v );
	~CCShape();
	void			SetDims( const int64_t* dims, unsigned int rank );
	unsigned int	GetDims( int64_t* dims, unsigned int rank ) const;
	void	Dump( const char* text= "" ) const;
	unsigned int	GetRank() const
	{
		return	Rank;
	}
	void	SetRank( unsigned int rank )
	{
		Rank= rank;
	}
	size_t	GetElementCount() const;
	int	Get( unsigned int index ) const
	{
		if( index < Rank ){
			return	Dimension[index];
		}
		return	1;
	}
	int	GetR( unsigned int index ) const
	{
		if( index < Rank ){
			return	Dimension[Rank - index -1];
		}
		return	1;
	}
	void	Set( unsigned int index, int dim )
	{
		if( index < Rank ){
			Dimension[index]= dim;
		}
	}
	void	SetR( unsigned int index, int dim )
	{
		if( index < Rank ){
			Dimension[Rank - index -1]= dim;
		}
	}
	int		GetX() const
	{
		return	Rank >= 1 ? Dimension[Rank-1] : 1;
	}
	int		GetY() const
	{
		return	Rank >= 2 ? Dimension[Rank-2] : 1;
	}
	int		GetZ() const
	{
		return	Rank >= 3 ? Dimension[Rank-3] : 1;
	}
	int		GetW() const
	{
		return	Rank >= 4 ? Dimension[Rank-4] : 1;
	}
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
}

#endif

