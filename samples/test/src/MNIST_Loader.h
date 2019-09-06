// twrapper 2019/09/01 Hiroyuki Ogasawara
// vim:ts=4 sw=4 noet:

#pragma once


class MNIST_Loader {
	enum : unsigned int {
		FILE_COUNT			=	4,
		HEADER_OFFSET_IMAGE	=	16,
		HEADER_OFFSET_LABEL	=	8,
		IMAGE_BYTE_SIZE		=	28*28,
		LABEL_COUNT			=	10,
	};
	void*	iImage[4];
public:
	MNIST_Loader();
	~MNIST_Loader();
	void	Finalize();
	void	Load();

	void			GetImage( float* buffer, unsigned int index, bool test );
	unsigned int	GetLabel( unsigned int index, bool test ) const;
	void			GetLabel10( float* buffer, unsigned int index, bool test );
	static unsigned int		ArgMax10( const float* buffer );
};




