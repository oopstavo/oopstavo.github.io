jQuery(document).ready(function() {
	var QRBox	=	$('#QRBox');
	var MainBox	=	$('#MainBox');
	var BTCQR	=	'/images/BTCQR.jpeg';	// 二维码路径
	var AliPayQR	=	'/images/AliPayQR.jpeg';
	var WeChatQR	=	'/images/WeChatQR.jpeg';


	$('#BTC').mouseenter(function()
	{
		MainBox.css('background-image','url('+BTCQR+')');
		QRBox.fadeIn(100)
	});
	$('#AliPay').mouseenter(function()
	{
		MainBox.css('background-image','url('+AliPayQR+')');
		QRBox.fadeIn(100)
	});
	$('#WeChat').mouseenter(function()
	{
		MainBox.css('background-image','url('+WeChatQR+')');
		QRBox.fadeIn(100)
	});

	$('#donateBox').mouseleave(function()
	{
	  	QRBox.fadeOut(300)
	});
});
